import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from './config.js';

const API_KEY = 'API_KEY';

interface Message {
  role: 'user' | 'model';
  content: string;
}

interface Completion {
  Content: string;
  TokenUsage?: number;
}

interface ErrorCompletion {
  Error: string;
}

type ConnectorResponse = {
  Completions: Array<Completion | ErrorCompletion>;
  ModelType?: string;
};

const mapErrorToCompletion = (error: unknown): ErrorCompletion => {
  const errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
  return {
    Error: errorMessage,
  };
};

async function main(
  model: string,
  prompts: string[],
  properties: Record<string, unknown>,
  settings: Record<string, unknown>,
): Promise<ConnectorResponse> {
  try {
    const { ...restProperties } = properties;

    const genAI = new GoogleGenerativeAI(settings?.[API_KEY] as string);
    const geminiModel = genAI.getGenerativeModel({
      model: model,
      ...restProperties
    });

    const outputs: Array<Completion | ErrorCompletion> = [];
    let chatHistory: Message[] = [];
    let chat = geminiModel.startChat({
      history: chatHistory.map((msg) => ({
        role: msg.role,
        parts: [{ text: msg.content }],
      })),
    });

    for (const prompt of prompts) {
      try {
        chatHistory.push({ role: 'user', content: prompt });
        const result = await chat.sendMessage(prompt);
        const response = result.response;
        const text = response.text();

        // Count tokens
        const { totalTokens } = await geminiModel.countTokens(prompt);

        chatHistory.push({ role: 'model', content: text });
        outputs.push({ Content: text, TokenUsage: totalTokens });
      } catch (error) {
        const completionWithError = mapErrorToCompletion(error);
        outputs.push(completionWithError);
      }
    }

    return {
      Completions: outputs,
    };
  } catch (error) {
    console.error('Error in main function:', error);
    return {
      Completions: [mapErrorToCompletion(error)],
    };
  }
}

export { main, config };
