package com.github.tjake.jlama.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.Tensor;
import com.google.common.collect.ImmutableMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class SafeTensorIndex implements WeightLoader, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(SafeTensorIndex.class);
    private static final ObjectMapper om = new ObjectMapper();

    private final Map<String, Object> metadata;

    // Map from weight name to file name (this is what's in the JSON file)
    private final Map<String, String> weightFileMap;

    // Map from weight name to Weights data
    private final Map<String, Weights> weightMap = new HashMap<>();


    // Map from file name to RandomAccessFile
    private final Map<String, RandomAccessFile> fileMap = new HashMap<>();

    public static SafeTensorIndex loadWithWeights2(Path modelRoot) throws IOException {
        SafeTensorIndex index = om.readValue(Paths.get(modelRoot.toString(), "model.safetensors.index.json").toFile(), SafeTensorIndex.class);

        for (Map.Entry<String, String> e : index.weightFileMap.entrySet()) {
            if (!index.fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                index.fileMap.put(e.getValue(), raf);
                long s = raf.length();
                long s2 = Integer.MAX_VALUE;
                if (s > s2)
                    throw new IllegalArgumentException("File too large: " + e.getValue());

                index.weightMap.put(e.getValue(), SafeTensors.readBytes(raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length())));
            }
        }

        return index;
    }

    public static SafeTensorIndex loadWithWeights(Path modelRoot) throws IOException {
        SafeTensorIndex index = om.readValue(Paths.get(modelRoot.toString(), "model.safetensors.index.json").toFile(), SafeTensorIndex.class);

        for (Map.Entry<String, String> e : index.weightFileMap.entrySet()) {
            // Only load the file if it's not already loaded
            if (!index.fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                index.fileMap.put(e.getValue(), raf);

                //Read the first 1MB of the file to get the TensorInfo
                ByteBuffer header = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, raf.length()));

                Map<String, String> metadata = new HashMap<>();
                Map<String, TensorInfo> tensorInfoMap = SafeTensors.readTensorInfoMap(header, Optional.of(metadata));
                int endOfHeaderPosition = header.position();

                Map<List<Long>, List<String>> splits = index.computeMmapSplits(tensorInfoMap, raf.length());
                for (Map.Entry<List<Long>, List<String>> split : splits.entrySet()) {
                    long offset = split.getKey().get(0);
                    long length = split.getKey().get(1);
                    List<String> tensors = split.getValue();

                    ByteBuffer buf = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, endOfHeaderPosition + offset, (length - offset));
                    Map<String, TensorInfo> mmapTensorInfoMap = tensorInfoMap.entrySet().stream()
                            .filter(x -> tensors.contains(x.getKey())).collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

                    Weights mmapWeights = new Weights(metadata, mmapTensorInfoMap, buf.slice());
                    for (String tensor : tensors) {
                        index.weightMap.put(tensor, mmapWeights);
                    }
                }
            }
        }

        return index;
    }

    /**
     * Group tensors into splits that can be mmaped together.
     * Since mmap limitation is integer max_value length.
     *
     * This also adjusts (inplace) the tensor offsets to be relative to the start of the split.
     *
     */
    private Map<List<Long>, List<String>> computeMmapSplits(Map<String, TensorInfo> tensorInfoMap, long fileLength) {
        Set<String> added = new HashSet<>();
        Map<List<Long>, List<String>> splits = new HashMap<>();
        long lastSplitOffset = 0;
        while (added.size() < tensorInfoMap.size()) {
            List<String> tensors = new ArrayList<>();
            long limit = lastSplitOffset + Integer.MAX_VALUE;
            long startOffset = fileLength;
            long endOffset = 0;

            for (Map.Entry<String, TensorInfo> e : tensorInfoMap.entrySet()) {
                if (added.contains(e.getKey()))
                    continue;

                TensorInfo info = e.getValue();

                if (info.dataOffsets[1] < limit) {
                    tensors.add(e.getKey());
                    added.add(e.getKey());

                    if (info.dataOffsets[1] > endOffset)
                        endOffset = info.dataOffsets[1];

                    if (info.dataOffsets[0] < startOffset)
                        startOffset = info.dataOffsets[0];

                    // Adjust the offset to be relative to the start of the split
                    info.dataOffsets[0] -= lastSplitOffset;
                    info.dataOffsets[1] -= lastSplitOffset;

                    logger.debug("Adding tensor {} to split {}-{}", e.getKey(), info.dataOffsets[0], info.dataOffsets[1]);
                }
            }

            logger.debug("Adding split {}-{} with {} tensors", startOffset, endOffset, tensors.size());
            assert endOffset - startOffset < Integer.MAX_VALUE : "Mmap split too large " + (endOffset - startOffset) + " > " + Integer.MAX_VALUE + " " + lastSplitOffset;
            splits.put(List.of(startOffset, endOffset), tensors);
            lastSplitOffset = endOffset;
        }

        return splits;
    }
    public Tensor load2(String name) {
        String f = weightFileMap.get(name);
        if (f == null)
            throw new NoSuchElementException(name);

        return weightMap.get(f).load(name);
    }

    public Tensor load(String name) {
        Weights w = weightMap.get(name);
        if (w == null)
            throw new NoSuchElementException(name);

        return w.load(name);
    }

    @JsonCreator
    SafeTensorIndex(@JsonProperty("metadata") Map<String, Object> metadata,
                           @JsonProperty("weight_map") Map<String, String> weightFileMap) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap);
    }

    @Override
    public void close() throws Exception {
        weightMap.clear();
        fileMap.forEach((k,v) -> {
            try {
                v.close();
            } catch (IOException e) {
                // Close quietly
            }
        });
        fileMap.clear();
    }
}