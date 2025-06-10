#!/usr/bin/env node

const { exec } = require('child_process');
const port = process.argv[2] || '3000';

console.log(`🔍 Checking for processes on port ${port}...`);

// Function to kill process on port (cross-platform)
function killPort(port) {
  return new Promise((resolve, reject) => {
    const isWindows = process.platform === 'win32';
    
    if (isWindows) {
      // Windows command
      exec(`netstat -ano | findstr :${port}`, (error, stdout) => {
        if (error) {
          console.log(`✅ Port ${port} is available`);
          resolve();
          return;
        }
        
        const lines = stdout.split('\n');
        const pids = new Set();
        
        lines.forEach(line => {
          const match = line.trim().match(/\s+(\d+)$/);
          if (match) {
            pids.add(match[1]);
          }
        });
        
        if (pids.size === 0) {
          console.log(`✅ Port ${port} is available`);
          resolve();
          return;
        }
        
        console.log(`💀 Killing ${pids.size} process(es) on port ${port}...`);
        const killPromises = Array.from(pids).map(pid => {
          return new Promise((killResolve) => {
            exec(`taskkill /PID ${pid} /F`, (killError) => {
              if (killError) {
                console.log(`⚠️  Could not kill PID ${pid}: ${killError.message}`);
              } else {
                console.log(`✅ Killed process ${pid}`);
              }
              killResolve();
            });
          });
        });
        
        Promise.all(killPromises).then(() => {
          console.log(`🎯 Port ${port} should now be available`);
          resolve();
        });
      });
    } else {
      // macOS/Linux command
      exec(`lsof -ti:${port}`, (error, stdout) => {
        if (error) {
          console.log(`✅ Port ${port} is available`);
          resolve();
          return;
        }
        
        const pids = stdout.trim().split('\n').filter(pid => pid);
        
        if (pids.length === 0) {
          console.log(`✅ Port ${port} is available`);
          resolve();
          return;
        }
        
        console.log(`💀 Killing ${pids.length} process(es) on port ${port}...`);
        
        const killCommand = `kill -9 ${pids.join(' ')}`;
        exec(killCommand, (killError) => {
          if (killError) {
            console.log(`⚠️  Error killing processes: ${killError.message}`);
            reject(killError);
          } else {
            console.log(`✅ Successfully killed processes: ${pids.join(', ')}`);
            console.log(`🎯 Port ${port} is now available`);
            resolve();
          }
        });
      });
    }
  });
}

// Execute the port killing
killPort(port)
  .then(() => {
    console.log(`🚀 Ready to start development server on port ${port}`);
    process.exit(0);
  })
  .catch((error) => {
    console.error(`❌ Failed to free port ${port}:`, error.message);
    process.exit(1);
  }); 