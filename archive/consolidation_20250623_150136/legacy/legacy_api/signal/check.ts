import type { NextApiRequest, NextApiResponse } from 'next';
import pool from '../../lib/db';

interface SignalQueryResult {
  rows: Array<{
    symbol: string;
    // Add other columns as needed
  }>;
}

interface ApiResponse {
  triggered: boolean;
  signal?: {
    symbol: string;
    // Add other columns as needed
  };
}

interface ApiError {
  error: string;
  details?: string;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse<ApiResponse | ApiError>) {
  try {
    const { ticker } = req.query;
    if (!ticker || typeof ticker !== 'string') {
      return res.status(400).json({ error: 'Missing or invalid ticker parameter' });
    }
    const result: SignalQueryResult = await pool.query(
      'SELECT * FROM signal WHERE symbol = $1 ORDER BY created_at DESC LIMIT 1',
      [ticker]
    );
    if (result.rows.length > 0) {
      res.status(200).json({ triggered: true, signal: result.rows[0] });
    } else {
      res.status(200).json({ triggered: false });
    }
  } catch (err) {
    res.status(500).json({ error: 'Server error', details: (err as Error).message });
  }
}
