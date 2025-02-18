package com.github.tjake.jlama.model;

import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.ToolCall;
import com.github.tjake.jlama.safetensors.prompt.ToolResult;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PromptBuilder {
  private final AbstractModel model;
  private final Map<String, List<String>> messages = new HashMap<>();
  private final List<ToolCall> toolCalls = new ArrayList<>();
  private final List<ToolResult> toolResults = new ArrayList<>();
  private boolean isGenerational = false;

  PromptBuilder(AbstractModel model) {
    this.model = model;
  }

  public PromptBuilder addUserMessage(String userMessage) {
    this.messages.computeIfAbsent("user", k -> new ArrayList<>()).add(userMessage);
    return this;
  }

  public PromptBuilder addSystemMessage(String systemMessage) {
    this.messages.computeIfAbsent("system", k -> new ArrayList<>()).add(systemMessage);
    return this;
  }

  public PromptBuilder addAssistantMessage(String assistantMessage) {
    this.messages.computeIfAbsent("assistant", k -> new ArrayList<>()).add(assistantMessage);
    return this;
  }

  public PromptBuilder addToolCall(ToolCall toolCall) {
    this.toolCalls.add(toolCall);
    return this;
  }

  public PromptBuilder addToolResult(ToolResult toolResult) {
    this.toolResults.add(toolResult);
    return this;
  }

  public PromptBuilder generationalPrompt(boolean isGenerational) {
    this.isGenerational = isGenerational;
    return this;
  }

  public PromptContext build() {
    PromptContext ctx;

    if (model.promptSupport().isPresent()) {
      var promptSupport = model.promptSupport().get().builder();

      if (this.messages.containsKey("user")) {
        this.messages.get("user").forEach(promptSupport::addUserMessage);
      }

      if (this.messages.containsKey("system")) {
        this.messages.get("system").forEach(promptSupport::addSystemMessage);
      }

      if (this.messages.containsKey("assistant")) {
        this.messages.get("assistant").forEach(promptSupport::addAssistantMessage);
      }

      if (!this.toolCalls.isEmpty()) {
        this.toolCalls.forEach(promptSupport::addToolCall);
      }

      if (!this.toolResults.isEmpty()) {
        this.toolResults.forEach(promptSupport::addToolResult);
      }

      promptSupport.addGenerationPrompt(isGenerational);

      ctx = promptSupport.build();
    } else {
      var userMessages = this.messages.get("user");
      if (userMessages.isEmpty()) throw new IllegalStateException("No user messages found");
      ctx = PromptContext.of(userMessages.get(0));
    }

    return ctx;
  }
}
