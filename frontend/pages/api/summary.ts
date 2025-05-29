import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export default async function handler(req: any, res: any) {
  const { signal } = req.body;
  const prompt = `Explain this trading signal to a retail trader:\n\nSignal: ${signal.type}\nConfidence: ${signal.confidence}%\nReason: ${signal.explanation}`;

  const completion = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: prompt }],
  });

  res.status(200).json({ summary: completion.choices[0].message.content });
}
