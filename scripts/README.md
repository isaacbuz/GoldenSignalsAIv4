# Infrastructure and Deployment Scripts

This directory contains scripts for deploying, managing, and maintaining GoldenSignalsAI infrastructure.

- `deploy.sh`: Unified deploy script (supports AWS, Azure, etc.)
- `migrate.sh`: Database migration utility
- `backup.sh`: Backup automation

## Usage
Make scripts executable and run as needed:
```bash
chmod +x deploy.sh
./deploy.sh --provider aws
```

## Notes
- Store only non-secret, reusable scripts here.
- Use environment variables or a vault for secrets.
