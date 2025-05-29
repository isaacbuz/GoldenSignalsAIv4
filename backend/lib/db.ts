// Dummy pool for compatibility with Jest tests and PostgreSQL API usage
// Replace with actual pg.Pool in production
const pool = {
  query: async (..._args: any[]) => {
    return { rows: [] };
  }
};

export default pool;
