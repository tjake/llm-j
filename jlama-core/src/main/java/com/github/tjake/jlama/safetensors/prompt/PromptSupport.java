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
package com.github.tjake.jlama.safetensors.prompt;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.github.tjake.jlama.safetensors.tokenizer.TokenizerModel;
import com.github.tjake.jlama.util.JsonSupport;
import com.hubspot.jinjava.Jinjava;
import com.hubspot.jinjava.JinjavaConfig;
import com.hubspot.jinjava.LegacyOverrides;
import com.hubspot.jinjava.interpret.RenderResult;
import com.hubspot.jinjava.lib.fn.ELFunctionDefinition;
import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class also renders the prompt templates of the huggingface model format (using jinja templates)
 * @see <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models">Chat Templating</a>
 */
public class PromptSupport {
    private static final Logger logger = LoggerFactory.getLogger(PromptSupport.class);

    // This matches the jinja config in huggingface
    private static final Jinjava jinjava = new Jinjava(
        JinjavaConfig.newBuilder()
            .withTrimBlocks(true)
            .withLstripBlocks(true)
            .withLegacyOverrides(
                LegacyOverrides.newBuilder()
                    .withParseWhitespaceControlStrictly(true)
                    .withUseTrimmingForNotesAndExpressions(true)
                    .withUseSnakeCasePropertyNaming(true)
                    .withKeepNullableLoopValues(true)
                    .build()
            )
            .withObjectMapper(
                new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)
                    .setDefaultPrettyPrinter(JsonSupport.JlamaPrettyPrinter.INSTANCE)
            )
            .build()
    );

    static {
        jinjava.getGlobalContext()
            .registerFunction(new ELFunctionDefinition("", "raise_exception", PromptSupport.class, "raiseException", String.class));
    }

    private final TokenizerModel m;

    public PromptSupport(TokenizerModel model) {
        this.m = model;
    }

    public Builder builder() {
        return new Builder(this.m);
    }

    public static void raiseException(String message) {
        logger.debug("Prompt template error: " + message);
    }

    private enum PromptType {
        DEFAULT,
        TOOL,
        RAG
    }

    private enum PromptRole {
        USER,
        SYSTEM,
        ASSISTANT,
        TOOL,
        TOOL_CALL
    }

    static class Message {
        private final Object content;
        private final PromptRole role;
        private final ToolCallFunction toolCalls;
        private final String toolCallId;

        private Message(Object content, PromptRole role) {
            this.content = content;
            this.role = role;
            this.toolCalls = null;
            this.toolCallId = null;
        }

        private Message(ToolCall toolCall) {
            this.content = null;
            this.role = PromptRole.TOOL_CALL;
            this.toolCalls = new ToolCallFunction(toolCall);
            this.toolCallId = toolCall.getId();
        }

        private Message(ToolResult toolResult) {
            this.content = toolResult.toJson();
            this.toolCalls = null;
            this.role = PromptRole.TOOL;
            this.toolCallId = toolResult.getToolCallId();
        }

        public Object getContent() {
            return content;
        }

        // Jinja template expects a map for each message
        public Map toMap() {
            Map map = new HashMap();
            map.put("role", role.name().toLowerCase());
            map.put("content", content == null ? "" : content);

            if (toolCalls != null) {
                map.put("tool_calls", List.of(toolCalls.toMap()));
            }

            if (toolCallId != null) {
                map.put("tool_call_id", toolCallId);
            }

            return map;
        }

        public String getRole() {
            return role.name().toLowerCase();
        }

        public List<ToolCallFunction> toolCalls() {
            if (toolCalls == null) {
                return null;
            }

            return List.of(toolCalls);
        }
    }

    static class ToolCallFunction {
        private final ToolCall call;

        private ToolCallFunction(ToolCall call) {
            this.call = call;
        }

        public InnerToolCall function() {
            return new InnerToolCall(call);
        }

        public Map toMap() {
            Map<String, Object> args = new LinkedHashMap<>();
            args.put("name", call.getName());
            args.put("arguments", call.getParameters());
            return Map.of("function", args, "id", call.getId());
        }
    }

    static class InnerToolCall {
        private final ToolCall call;

        private InnerToolCall(ToolCall call) {
            this.call = call;
        }

        public Map<String, Object> arguments() {
            return call.getParameters();
        }

        public String name() {
            return call.getName();
        }
    }

    public static class Builder {
        private final TokenizerModel m;
        private PromptType type = PromptType.DEFAULT;
        private boolean addGenerationPrompt = true;

        private final List<Message> messages = new ArrayList<>(2);
        private boolean stripPreamble = false;

        private Builder(TokenizerModel m) {
            this.m = m;
        }

        public Builder usePromptType(PromptType type) {
            this.type = type;
            return this;
        }

        public Builder addGenerationPrompt(boolean addGenerationPrompt) {
            this.addGenerationPrompt = addGenerationPrompt;
            return this;
        }

        public Builder addUserMessage(String content) {
            messages.add(new Message(content, PromptRole.USER));
            return this;
        }

        public Builder addToolResult(ToolResult result) {
            messages.add(new Message(result));
            return this;
        }

        public Builder addToolCall(ToolCall call) {
            messages.add(new Message(call));
            return this;
        }

        public Builder addSystemMessage(String content) {
            messages.add(new Message(content, PromptRole.SYSTEM));
            return this;
        }

        public Builder addAssistantMessage(String content) {
            messages.add(new Message(content, PromptRole.ASSISTANT));
            return this;
        }

        public Builder stripPreamble() {
            stripPreamble = true;
            return this;
        }

        public PromptContext build() {
            return build(Optional.empty());
        }

        public PromptContext build(List<Tool> tools) {
            return build(Optional.of(tools));
        }

        public PromptContext build(Tool... tools) {
            return build(Optional.of(List.of(tools)));
        }

        private PromptContext build(Optional<List<Tool>> optionalTools) {
            if (messages.isEmpty()) {
                throw new IllegalArgumentException("No messages to generate prompt");
            }

            if (m.promptTemplates().isEmpty()) {
                throw new UnsupportedOperationException("Prompt templates are not available for this model");
            }

            String template = m.promptTemplates()
                .map(t -> t.get(type.name().toLowerCase()))
                .orElseThrow(() -> new UnsupportedOperationException("Prompt template not available for type: " + type));

            if (optionalTools.isPresent() && !optionalTools.get().isEmpty() && !m.hasToolSupport()) logger.warn(
                "This model does not support tools, but tools are specified"
            );


            String preamble = "";
            if (stripPreamble) {
                Map<String, Object> args = new HashMap<>();
                args.putAll(
                    Map.of(
                        "messages",
                        Map.of(),
                        "add_generation_prompt",
                        false,
                        "eos_token",
                        m.eosToken(),
                        "bos_token",
                        ""
                    )
                ); // We add the BOS ourselves
                optionalTools.ifPresent(tools -> args.put("tools", tools));

                RenderResult r = jinjava.renderForResult(template, args);
                preamble = r.getOutput();
            }

            Map<String, Object> args = new HashMap<>();
            args.putAll(
                Map.of(
                    "messages",
                    messages.stream().map(Message::toMap).toList(),
                    "add_generation_prompt",
                    addGenerationPrompt,
                    "eos_token",
                    m.eosToken(),
                    "bos_token",
                    ""
                )
            ); // We add the BOS ourselves

            optionalTools.ifPresent(tools -> args.put("tools", tools));

            RenderResult r = jinjava.renderForResult(template, args);

            if (r.hasErrors()) logger.debug("Prompt template errors: " + r.getErrors());

            String output = r.getOutput();
            return new PromptContext(output.substring(preamble.length()), optionalTools);
        }
    }
}
