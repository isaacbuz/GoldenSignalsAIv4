import { promises as fs } from 'fs';
import path from 'path';

const RULES_PATH = path.join(process.cwd(), 'signal-rules.json');

export default async function handler(req: any, res: any) {
  if (req.method === 'GET') {
    try {
      const data = await fs.readFile(RULES_PATH, 'utf8');
      res.status(200).json(JSON.parse(data));
    } catch (e) {
      res.status(200).json({});
    }
  } else if (req.method === 'POST') {
    await fs.writeFile(RULES_PATH, req.body);
    res.status(200).json({ ok: true });
  } else {
    res.status(405).json({ message: 'Method not allowed' });
  }
}
