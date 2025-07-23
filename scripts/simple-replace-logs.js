#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Function to recursively find all TypeScript files
function findFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      // Skip certain directories
      if (!file.startsWith('.') && file !== 'node_modules' && file !== 'dist' && file !== 'build') {
        findFiles(filePath, fileList);
      }
    } else if (file.endsWith('.ts') || file.endsWith('.tsx')) {
      // Skip test files and the logger itself
      if (!file.includes('.test.') && !file.includes('.spec.') && !filePath.includes('logger.ts')) {
        fileList.push(filePath);
      }
    }
  });

  return fileList;
}

// Main function
function main() {
  const frontendSrcPath = path.join(__dirname, '..', 'frontend', 'src');
  const files = findFiles(frontendSrcPath);

  console.log(`Found ${files.length} files to process`);

  let processedCount = 0;

  files.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;

    // Check if file uses console methods
    if (content.includes('console.')) {
      // Replace console methods
      const originalContent = content;
      content = content.replace(/console\.log\(/g, 'logger.info(');
      content = content.replace(/console\.error\(/g, 'logger.error(');
      content = content.replace(/console\.warn\(/g, 'logger.warn(');
      content = content.replace(/console\.info\(/g, 'logger.info(');
      content = content.replace(/console\.debug\(/g, 'logger.debug(');

      modified = content !== originalContent;

      if (modified) {
        // Check if logger is already imported
        const hasLoggerImport = content.includes("from '../services/logger'") ||
                               content.includes('from "./services/logger"') ||
                               content.includes("from '@/services/logger'") ||
                               content.includes('logger from');

        if (!hasLoggerImport) {
          // Calculate relative path to logger
          const relativePath = path.relative(path.dirname(filePath), path.join(frontendSrcPath, 'services', 'logger'))
            .replace(/\\/g, '/');

          // Add import statement
          const importStatement = `import logger from '${relativePath.startsWith('.') ? relativePath : './' + relativePath}';\n`;

          // Find where to insert the import
          const lines = content.split('\n');
          let insertIndex = 0;

          // Find the last import statement
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim().startsWith('import ')) {
              insertIndex = i + 1;
            }
          }

          // Insert the import
          lines.splice(insertIndex, 0, importStatement);
          content = lines.join('\n');
        }

        // Write the file
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`✅ Processed: ${path.relative(process.cwd(), filePath)}`);
        processedCount++;
      }
    }
  });

  console.log(`\n✨ Complete! Processed ${processedCount} files`);
}

// Run the script
try {
  main();
} catch (error) {
  console.error('Error:', error);
  process.exit(1);
}
