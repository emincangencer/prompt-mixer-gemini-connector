import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from './config.js';

const API_KEY = 'API_KEY';

interface Message {
  role: string;
  content: string;
}

interface Completion {
  Content: string | null;
  TokenUsage: number | undefined;
}

interface ConnectorResponse {
  Completions: Completion[];
  ModelType: string;
}

interface GeminiError {
  error: {
    message: string;
  };
}

const mapToResponse = (outputs: Array<Completion>, model: string): ConnectorResponse => {
  return {
    Completions: outputs,
    ModelType: model,
  };
};

const mapErrorToCompletion = (error: GeminiError, model: string): Message => {
  const errorMessage = error.error.message;
  return { role: 'model', content: errorMessage };
};

async function main(
  model: string,
  prompts: string[],
  properties: Record<string, unknown>,
  settings: Record<string, unknown>,
) {
  const genAI = new GoogleGenerativeAI(settings['API_KEY'] as string);
  const geminiModel = genAI.getGenerativeModel({ model: model });
  const outputs: Completion[] = [];
  let chatHistory: Message[] = [];
  let chat = geminiModel.startChat({
    history: chatHistory.map((msg) => ({
      role: msg.role as 'user' | 'model',
      parts: msg.content,
    })),
    generationConfig: {
      maxOutputTokens: properties['maxOutputTokens'] as number,
      temperature: properties['temperature'] as number,
      topP: properties['topP'] as number,
      topK: properties['topK'] as number,
    },
  });

  try {
    for (let prompt of prompts) {
      try {
        chatHistory.push({ role: 'user', content: prompt });
        const result = await chat.sendMessage(prompt);
        const response = await result.response;
        const text = await response.text();
        chatHistory.push({ role: 'model', content: text });
        outputs.push({ Content: text, TokenUsage: undefined });

        console.log(`Response:`, text);
      } catch (error) {
        const completionWithError = mapErrorToCompletion(error as GeminiError, model);
        chatHistory.push(completionWithError);
      }
    }

    return mapToResponse(outputs, model);
  } catch (error) {
    console.error('Error in main function:', error);
    return { Error: error, ModelType: model };
  }
}

export { main, config };
