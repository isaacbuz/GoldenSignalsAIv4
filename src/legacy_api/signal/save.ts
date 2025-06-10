import type { NextApiRequest, NextApiResponse } from 'next';
import pool from '../../lib/db';

interface SignalData {
  symbol: string;
  [key: string]: any;
}

interface ErrorResponse {
  error: string;
  details?: string;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse<SignalData | ErrorResponse>) {
  if (req.method === 'POST') {
    try {
      const { symbol, ...rest } = req.body as SignalData;
      if (!symbol || typeof symbol !== 'string') {
        return res.status(400).json({ error: 'Missing or invalid symbol in request body' });
      }
      if (Object.keys(rest).length === 0) {
        return res.status(400).json({ error: 'Missing signal data in request body' });
      }
      const result = await pool.query(
        'INSERT INTO signal (symbol, data) VALUES ($1, $2) RETURNING *',
        [symbol, rest]
      );
      if (!result.rows[0]) {
        throw new Error('Failed to save signal');
      }
      res.status(201).json(result.rows[0]);
    } catch (err) {
      res.status(500).json({ error: 'Failed to save signal', details: (err as Error).message });
    }
  } else {
    res.status(405).json({ message: 'Method not allowed' });
  }
}
