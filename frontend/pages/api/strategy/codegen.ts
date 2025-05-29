import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export default async function handler(req: any, res: any) {
  const { text } = req.body;
  const prompt = `\nConvert this trading strategy to a JSON-based signal rule object.\n\nStrategy:\n"${text}"\n\nFormat:\n{\n  "Strategy Name": {\n    "indicators": [...],\n    "conditions": { ... },\n    "confidence": number\n  }\n}\nOnly return the JSON object.`;

  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: prompt }],
  });

  const json = response.choices[0].message.content;
  res.status(200).json({ json });
}
