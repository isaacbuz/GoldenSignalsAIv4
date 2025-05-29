#!/bin/bash
# This script sets up a cron job to run retrain_pretrained_models.py daily at 3am
CRON_JOB="0 3 * * * cd $(pwd)/backend/scripts && /usr/bin/env python3 retrain_pretrained_models.py >> $(pwd)/backend/scripts/retrain.log 2>&1"
# Write out current crontab, add new job, and install
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
echo "Retraining cron job installed."
