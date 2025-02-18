/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.model;

import static com.github.tjake.jlama.util.DebugSupport.debug;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.functions.*;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import com.github.tjake.jlama.safetensors.prompt.Tool;
import com.github.tjake.jlama.safetensors.prompt.ToolCall;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.tensor.*;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.DebugSupport;
import com.github.tjake.jlama.util.JsonSupport;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import jdk.incubator.vector.FloatVector;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class AbstractModel implements Generator {
    private static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    private static final Integer MAX_BATCH_SIZE = Integer.getInteger("jlama.max_batch_size", 256);

    public enum InferenceType {
        // Used for distributed inference
        INPUT_TO_EMBEDDING(true, false, false, false, false),
        OUTPUT_TO_TOKEN(false, false, true, false, false),
        FORWARD_PASS(true, true, false, false, false),

        // Used for different types of inference
        FULL_GENERATION(true, true, true, false, false),
        FULL_CLASSIFICATION(true, true, false, true, true),
        FULL_EMBEDDING(true, true, false, false, true);

        final boolean isInput;
        final boolean isOutput;
        final boolean isClassify;
        final boolean isFwdPass;
        final boolean isPooling;

        InferenceType(boolean isInput, boolean isFwdPass, boolean isOutput, boolean isClassify, boolean isPooling) {
            this.isInput = isInput;
            this.isOutput = isOutput;
            this.isFwdPass = isFwdPass;
            this.isClassify = isClassify;
            this.isPooling = isPooling;
        }
    }

    protected final InferenceType inferenceType;
    protected final Config c;
    protected final WeightLoader weights;
    protected final Tokenizer tokenizer;
    protected final DType modelDType;
    protected final DType workingDType;
    protected final DType workingQType;
    protected final Optional<DType> modelQType;
    protected EmbedInput embedInput;
    protected SampleOutput sampleOutput;
    protected ClassifyOutput classifyOutput;
    protected Optional<PoolingLayer> poolingLayer;
    protected TransformerBlock[] transformerBlocks;
    protected KvBufferCache kvBufferCache;

    protected AbstractModel(
        InferenceType inferenceType,
        Config c,
        WeightLoader w,
        Tokenizer t,
        DType workingMemoryDType,
        DType workingMemoryQType,
        Optional<DType> modelQType
    ) {
        this.inferenceType = inferenceType;
        this.c = c;
        this.weights = w;
        this.tokenizer = t;

        this.modelDType = w.getModelDType();
        this.workingDType = workingMemoryDType;
        this.modelQType = modelQType;
        this.kvBufferCache = new KvBufferCache(this);

        // FIXME: This is a hack to support Avoid Q8F32 evals
        if (modelDType == DType.F32 && workingMemoryQType != DType.F32 && modelQType.isEmpty()) {
            workingMemoryQType = DType.F32;
        }

        // FIXME: This is a hack to support Avoid Q8BF16 evals
        if (modelDType == DType.BF16 && workingMemoryQType != DType.BF16 && modelQType.isEmpty()) {
            workingMemoryQType = DType.BF16;
        }

        // Check to make sure the model is big enough to support Q4I8 computations
        // If not, fall back to F32
        if (modelDType == DType.Q4
            && workingMemoryQType == DType.I8
            && ((c.embeddingLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0
                || (c.hiddenLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0)) {
            workingMemoryQType = DType.F32;
        }

        // Check to make sure the model is big enough to support Q4I8 computations
        // If not, fall back to F32
        if (modelDType == DType.Q4
            && workingMemoryQType == DType.I8
            && (c.embeddingLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0) {
            workingMemoryQType = DType.F32;
        }

        if (workingMemoryQType != workingMemoryDType) {
            boolean supportsQType;
            AbstractTensor tmp = makeDenseTensor(Q8ByteBufferTensor.BLOCK_SIZE);
            try (AbstractTensor tmp2 = TensorOperationsProvider.get().quantize(tmp, workingMemoryQType, 0, Q8ByteBufferTensor.BLOCK_SIZE)) {
                supportsQType = tmp2.dType() == workingMemoryQType;
                if (!supportsQType) {
                    logger.warn("Quantized memory type {} not supported, falling back to {}", workingMemoryQType, workingMemoryDType);
                    this.workingQType = this.workingDType;
                } else {
                    this.workingQType = workingMemoryQType;
                }
            }
        } else {
            this.workingQType = workingMemoryQType;
        }

        logger.info(
            "Model type = {}, Working memory type = {}, Quantized memory type = {}",
            this.modelDType,
            this.workingDType,
            this.workingQType
        );

        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
        this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
    }

    @Override
    public void close() {
        kvBufferCache.close();
    }

    protected abstract EmbedInput loadInputWeights();

    protected abstract TransformerBlock[] loadTransformerBlockWeights();

    protected abstract SampleOutput loadOutputWeights();

    protected ClassifyOutput loadClassifierWeights() {
        throw new UnsupportedOperationException("Classification not supported by this model");
    }

    protected PoolingLayer loadPoolingWeights() {
        return null;
    }

    public abstract ModelSupport.ModelType getModelType();

    public InferenceType getInferenceType() {
        return inferenceType;
    }

    public DType getWorkingDType() {
        return workingDType;
    }

    public Config getConfig() {
        return c;
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public WeightLoader getWeights() {
        return weights;
    }

    public Optional<PromptSupport> promptSupport() {
        return tokenizer.promptSupport();
    }

    public PromptBuilder prompt() {
        return new PromptBuilder(this);
    }

    public AbstractTensor makeTensor(int... shape) {
        TensorShape s = TensorShape.of(shape);
        return c.tensorCache.get(workingDType, s);
    }

    public AbstractTensor makeDenseTensor(int... shape) {
        return c.tensorCache.get(workingDType, TensorShape.of(shape));
    }

    public AbstractTensor makeDenseTensor(TensorShape s) {
        return c.tensorCache.get(workingDType, s);
    }

    protected AbstractTensor maybeQuantize(AbstractTensor t) {
        AbstractTensor t2 = c.tensorCache.get(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
    }

    protected AbstractTensor forward(int token_id, int pos, KvBufferCache.KvBuffer kvbuf) {
        return forward(token_id, pos, kvbuf, Optional.empty());
    }

    /**
     * This is a distributed version of forward pass that serves as a coordination point for the
     * distributed model.  The layers are split into one or more heads and each head is processed
     * by a different node.
     *
     * @param token_id
     * @param pos
     * @param kvbuf
     * @return
     */
    public AbstractTensor forward(
        int token_id,
        int pos,
        KvBufferCache.KvBuffer kvbuf,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        AbstractTensor embedding = embedInput.inputTokenToEmbedding(token_id, pos);

        debug("EMBEDDING TOKEN", token_id);
        debug("TOKEN POSITION", pos);

        return forward(embedding, pos, kvbuf, tensorReducer);
    }

    protected AbstractTensor batchForwardSlow(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        AbstractTensor last = null;
        for (int i = 0; i < token_ids.length; i++) {
            if (last != null) last.close();
            last = forward(token_ids[i], startPos + i, kvbuf);
        }

        return last;
    }

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(token_ids, startPos, kvbuf, Optional.empty());
    }

    public AbstractTensor batchForward(
        int[] token_ids,
        int startPos,
        KvBufferCache.KvBuffer kvbuf,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        AbstractTensor embedding = null;

        //Batch prompt into groups of MAX_BATCH_SIZE
        for (int i = 0; i < token_ids.length; i += MAX_BATCH_SIZE) {
            int[] batch = Arrays.copyOfRange(token_ids, i, Math.min(token_ids.length, i + MAX_BATCH_SIZE));
            embedding = embedInput.batchInputsToEmbeddings(batch, startPos + i);
            embedding = forward(embedding, startPos + i, kvbuf, tensorReducer);
            logger.debug("Batched forward pass for tokens {} to {}", i, i + batch.length);
        }

        return embedding;
    }

    public AbstractTensor forward(
        AbstractTensor embedding,
        int startPos,
        KvBufferCache.KvBuffer kvbuf,
        Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {

        for (int i = c.dctx().layerStart; i < c.dctx().layerEnd; i++) {
            int relativeLayer = i - c.dctx().layerStart;
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[relativeLayer].forward(embedding, startPos, kvbuf, tensorReducer);
            ref.close();
        }

        return embedding;
    }

    @Override
    public float[] embed(String input, PoolingType poolingType) {
        int[] encoded = Arrays.stream(tokenizer.encode(input)).mapToInt(Ints::checkedCast).toArray();

        Preconditions.checkArgument(encoded.length < c.contextLength);
        float[] outputEmbedding = new float[c.embeddingLength];

        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getEphemeralKvBuffer()) {
            int promptLength = encoded.length;
            float avgp = 1.0f / promptLength;

            try (AbstractTensor r = batchForward(encoded, 0, kvmem)) {
                if (poolingType == PoolingType.MODEL) {
                    if (poolingLayer.isPresent()) {

                        // Get the last value should represent the sum of the prompt (due to attention)
                        AbstractTensor output = r.slice(promptLength - 1);
                        AbstractTensor pooled = makeDenseTensor(1, c.embeddingLength);

                        // Pooling
                        TensorOperationsProvider.get()
                            .batchDotProduct(pooled, output, poolingLayer.get().getPoolingWeights(), 0, 0, c.embeddingLength);

                        poolingLayer.get()
                            .getPoolingBias()
                            .ifPresent(bias -> { TensorOperationsProvider.get().accumulate(pooled, bias, 0, c.embeddingLength); });

                        VectorMath.pfor(0, c.embeddingLength, i -> {
                            // BERT seems to use tanh for pooling rather than gelu
                            outputEmbedding[i] = ActivationFunction.eval(ActivationFunction.Type.TANH, pooled.get(0, i));
                        });

                        return outputEmbedding;
                    }

                    throw new UnsupportedOperationException("Pooling layer not found");
                }

                // No pooling layer, so we just pool manually embeddings
                for (int i = 0; i < promptLength; i++) {
                    AbstractTensor output = r.slice(i);
                    // Pooling
                    for (int ii = 0; ii < c.embeddingLength; ii++) {
                        switch (poolingType) {
                            case AVG:
                                outputEmbedding[ii] += output.get(0, ii) * avgp;
                                break;
                            case MAX:
                                outputEmbedding[ii] = Math.max(outputEmbedding[ii], output.get(0, ii));
                                break;
                            case SUM:
                                outputEmbedding[ii] += output.get(0, ii);
                                break;
                        }
                    }
                }
            }
            VectorMath.l2normalize(outputEmbedding);
            return outputEmbedding;
        }
    }

    @Override
    public Map<String, Float> classify(String input, PoolingType poolingType) {
        if (!c.isClassifier() || classifyOutput == null) {
            throw new UnsupportedOperationException("Classification not supported by this model");
        }

        float[] embedding = embed(input, poolingType);
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.wrap(embedding), TensorShape.of(embedding.length), false);

        int classes = classifyOutput.getClassificationWeights().shape().first();
        AbstractTensor scores = makeDenseTensor(classes);

        TensorOperationsProvider.get().batchDotProduct(scores, b, classifyOutput.getClassificationWeights(), 0, 0, c.embeddingLength);

        classifyOutput.getClassificationBias().ifPresent(bias -> { TensorOperationsProvider.get().accumulate(scores, bias, 0, classes); });

        VectorMath.softMax(scores, 0, classes);
        Map<String, Float> result = new HashMap<>();
        for (int i = 0; i < classes; i++) {
            String label = c.classifcationLabels.get().inverse().get(i);
            Float score = scores.get(0, i);

            result.put(label, score);
        }

        return result;
    }

    public float[] getLogits(AbstractTensor output) {
        try (
            AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output);
            AbstractTensor logits = makeDenseTensor(1, c.vocabularySize)
        ) {

            VectorMath.pchunk(0, c.vocabularySize, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0, c.embeddingLength, chunkStart, chunkSize);
            });

            VectorMath.softMax(logits, 0, c.vocabularySize);

            float[] r = new float[c.vocabularySize];

            // Convert from Tensor to float array
            logits.getMemorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(r);

            return r;
        }
    }

    public int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output)) {
            // This is a mix of argmax and sampling with softmax
            VectorMath.pchunk(0, c.vocabularySize, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get()
                    .dotProductChunk(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0, c.embeddingLength, chunkStart, chunkSize);
            });

            if (c.logitMultiplier != null) {
                TensorOperationsProvider.get().scale(1.0f / c.logitMultiplier, logits, 0, c.vocabularySize);
            }

            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (c.finalLogitSoftCapping != null) {
                    v /= c.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * c.finalLogitSoftCapping;
                    logits.set(v, 0, i);
                }
                if (v > maxv) {
                    maxi = i;
                    maxv = v;
                }
            }

            if (temperature == 0.0) {
                return maxi;
            }

            float sum = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = (float) FastMath.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }

            float acc = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) return i;
            }

            return c.vocabularySize - 1;
        }
    }

    protected boolean addBosToken() {
        return true;
    }

    public int[] encodePrompt(PromptContext promptContext) {
        long[] encoded = tokenizer.encode(promptContext.getPrompt());

        if (!addBosToken()) return Arrays.stream(encoded).mapToInt(Ints::checkedCast).toArray();

        // Remove BOS token if it's the first token, we explicitly add it below
        if (encoded.length > 0 && encoded[0] == c.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }

        int[] promptTokens = new int[(1 + encoded.length)];
        promptTokens[0] = c.bosToken;
        for (int i = 1; i <= encoded.length; i++)
            promptTokens[i] = Ints.checkedCast(encoded[i - 1]);

        return promptTokens;
    }

    @Override
    public Response generate(
        UUID sessionId,
        PromptContext promptContext,
        float temperature,
        int ntokens,
        BiConsumer<String, Float> onTokenWithTimings
    ) {
        long[] encoded = tokenizer.encode(promptContext.getPrompt());

        // Remove BOS token if it's the first token, we explicitly add it below
        if (encoded.length > 0 && encoded[0] == c.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }

        Preconditions.checkArgument(encoded.length < c.contextLength && encoded.length < ntokens, "Prompt exceeds max tokens");

        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(sessionId)) { // k and v for context window
            int startPos = kvmem.getCurrentContextPosition(); // Number of tokens in the buffer

            logger.debug("Starting at token {} for session {} with prompt {}", startPos, sessionId, promptContext.getPrompt());

            if (ntokens > c.contextLength) ntokens = c.contextLength;

            FinishReason reason = FinishReason.MAX_TOKENS;
            int promptLength;
            long promptBatchTime;
            int tokensGenerated;
            StringBuilder responseText = new StringBuilder();
            StringBuilder responseTextWithSpecialTokens = new StringBuilder();

            try (AbstractTensor logits = makeDenseTensor(c.vocabularySize)) {
                int[] promptTokens;

                if (addBosToken()) {
                    promptTokens = new int[(1 + encoded.length)];

                    promptTokens[0] = c.bosToken;
                    for (int i = 1; i <= encoded.length; i++)
                        promptTokens[i] = Ints.checkedCast(encoded[i - 1]);
                    promptLength = encoded.length;
                } else {
                    promptTokens = Arrays.stream(encoded).mapToInt(Ints::checkedCast).toArray();
                    promptLength = encoded.length;
                }

                long start = System.currentTimeMillis();
                long promptStart = start;
                // Batch Process Prompt
                AbstractTensor last = DebugSupport.isDebug()
                    ? batchForwardSlow(promptTokens, startPos, kvmem)
                    : batchForward(promptTokens, startPos, kvmem);

                promptBatchTime = System.currentTimeMillis() - start;
                float batchMsPerToken = Math.round((((double) promptBatchTime) / (double) promptLength));
                logger.debug("{} prompt tokens in {}ms | {}ms per token", promptLength, promptBatchTime, batchMsPerToken);

                float genMsPerToken = 0;
                tokensGenerated = 0;
                int next = sample(last.slice(last.shape().first() - 1), temperature, ThreadLocalRandom.current().nextFloat(), logits);
                last.close();
                try {
                    String c = tokenizer.decode(next);
                    if (tokenizer.getModel().isSpecialToken(next)) {
                        responseTextWithSpecialTokens.append(c);
                    } else {
                        onTokenWithTimings.accept(c, batchMsPerToken);
                        responseText.append(c);
                        responseTextWithSpecialTokens.append(c);
                    }
                } catch (Exception e) {
                    logger.error("Failed to decode token {}", next, e);
                }

                start = System.currentTimeMillis();
                for (int i = startPos + promptTokens.length; i < ntokens; i++) {
                    AbstractTensor output = forward(next, i, kvmem);
                    tokensGenerated++;

                    next = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);

                    if (logger.isTraceEnabled()) logger.trace("Sampled token {} with temperature {}", next, temperature);
                    output.close();

                    kvmem.incrementContextPosition();

                    // Model may tell us it's done
                    if (c.eosTokens.contains(next)) {
                        reason = FinishReason.STOP_TOKEN;
                        break;
                    }

                    try {
                        String c = tokenizer.decode(next);

                        if (tokenizer.getModel().isSpecialToken(next)) {
                            responseTextWithSpecialTokens.append(c);
                        } else {
                            genMsPerToken = (System.currentTimeMillis() - start) / (float) (tokensGenerated);
                            onTokenWithTimings.accept(c, genMsPerToken);
                            responseTextWithSpecialTokens.append(c);
                            responseText.append(c);
                        }
                    } catch (Exception e) {
                        logger.error("Failed to decode token {}", next, e);
                    }
                }

                long end = System.currentTimeMillis();

                Response response = new Response(
                    responseText.toString(),
                    responseTextWithSpecialTokens.toString(),
                    reason,
                    promptLength,
                    tokensGenerated,
                    promptBatchTime,
                    end - start
                );
                logger.debug(
                    String.format(
                        "\n\nelapsed: %ds, prompt %.1fms per token, gen %.1fms per token\n",
                        TimeUnit.MILLISECONDS.toSeconds(end - promptStart),
                        batchMsPerToken,
                        genMsPerToken
                    )
                );

                return postProcessResponse(promptContext, response);
            }
        }
    }

    /**
     * This is a hook for subclasses to post process the response before returning it to the caller.
     * For example this can be used to handle tool calls.
     * @param response
     * @return */
    protected Generator.Response postProcessResponse(PromptContext promptContext, Generator.Response response) {
        if (!tokenizer.getModel().hasToolSupport() || !promptContext.hasTools() || response.finishReason != FinishReason.STOP_TOKEN) {
            return response;
        }

        // Look for a tool call based on the tool names
        List<Tool> tools = promptContext.getTools().get();
        boolean foundTool = false;
        for (Tool tool : tools) {
            if (response.responseTextWithSpecialTokens.contains(tool.getFunction().getName())) {
                foundTool = true;
                break;
            }
        }

        if (!foundTool) {
            return response;
        }

        try {
            // If we found a tool call, we need to extract the tool call from the response
            List<String> jsonCalls = JsonSupport.extractJsonFromString(response.responseText);
            if (jsonCalls.isEmpty()) {
                logger.warn("Tool call detected but no tool call found in response: {}", response.responseText);
                return response;
            }

            logger.debug("Found tool calls: {}", jsonCalls);
            List<ToolCall> toolCalls = new ArrayList<>(jsonCalls.size());
            for (String jsonCall : jsonCalls) {
                if (jsonCall.startsWith("[")) {
                    List<ToolCall> toolCallList = JsonSupport.om.readValue(jsonCall, new TypeReference<>() {});
                    toolCalls.addAll(toolCallList);
                } else {
                    ToolCall toolCall = JsonSupport.om.readValue(jsonCall, ToolCall.class);
                    toolCalls.add(toolCall);
                }
            }

            // Remove duplicates
            toolCalls = toolCalls.stream().sorted(Comparator.comparing(ToolCall::getName)).distinct().collect(Collectors.toList());

            for (int i = 0; i < toolCalls.size(); i++) {
                // Standard is to use 9 digit ids
                toolCalls.get(i).setId(String.format("%09d", i));
            }

            return response.copyWithToolCalls(toolCalls);
        } catch (JsonProcessingException e) {
            logger.error("Failed to parse tool call from response: {}", response.responseText, e);
        }

        return response;
    }
}
