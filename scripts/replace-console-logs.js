#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { glob } = require('glob');
const { minimatch } = require('minimatch');

// Patterns to replace
const replacements = [
  // console.log -> logger.info
  { pattern: /console\.log\(/g, replacement: 'logger.info(' },
  // console.error -> logger.error
  { pattern: /console\.error\(/g, replacement: 'logger.error(' },
  // console.warn -> logger.warn
  { pattern: /console\.warn\(/g, replacement: 'logger.warn(' },
  // console.info -> logger.info
  { pattern: /console\.info\(/g, replacement: 'logger.info(' },
  // console.debug -> logger.debug
  { pattern: /console\.debug\(/g, replacement: 'logger.debug(' }
];

// Files to process
const filePatterns = [
  'frontend/src/**/*.ts',
  'frontend/src/**/*.tsx',
  '!frontend/src/services/logger.ts', // Exclude the logger file itself
  '!frontend/src/**/*.test.ts',
  '!frontend/src/**/*.test.tsx',
  '!frontend/src/**/*.spec.ts',
  '!frontend/src/**/*.spec.tsx'
];

async function processFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf8');
    let modified = false;

    // Check if file already imports logger
    const hasLoggerImport = content.includes("from '../services/logger'") ||
                           content.includes('from "../services/logger"') ||
                           content.includes("from '@/services/logger'") ||
                           content.includes('from "../../services/logger"') ||
                           content.includes('from "../../../services/logger"');

    // Check if file uses console methods
    const usesConsole = replacements.some(({ pattern }) => pattern.test(content));

    if (usesConsole) {
      // Apply replacements
      replacements.forEach(({ pattern, replacement }) => {
        if (pattern.test(content)) {
          content = content.replace(pattern, replacement);
          modified = true;
        }
      });

      if (modified && !hasLoggerImport) {
        // Calculate relative path to logger
        const relativePath = path.relative(path.dirname(filePath), 'frontend/src/services/logger.ts')
          .replace(/\.ts$/, '')
          .replace(/\\/g, '/');

        // Add logger import at the top of the file
        const importStatement = `import logger from '${relativePath.startsWith('.') ? relativePath : './' + relativePath}';\n`;

        // Find the right place to insert the import
        const firstImportMatch = content.match(/^import .* from ['"].*['"];?$/m);
        if (firstImportMatch) {
          // Add after the last import
          const lines = content.split('\n');
          let lastImportIndex = 0;
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].match(/^import .* from ['"].*['"];?$/)) {
              lastImportIndex = i;
            }
          }
          lines.splice(lastImportIndex + 1, 0, importStatement);
          content = lines.join('\n');
        } else {
          // Add at the beginning of the file
          content = importStatement + '\n' + content;
        }
      }

      if (modified) {
        await fs.writeFile(filePath, content, 'utf8');
        console.log(`✅ Processed: ${filePath}`);
        return 1;
      }
    }

    return 0;
  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
    return 0;
  }
}

async function main() {
  console.log('🔍 Finding files to process...');

  // Get all files matching the patterns
  const files = [];
  for (const pattern of filePatterns) {
    if (pattern.startsWith('!')) continue;
    const matches = await glob(pattern);
    files.push(...matches);
  }

  // Filter out excluded patterns
  const excludePatterns = filePatterns.filter(p => p.startsWith('!')).map(p => p.slice(1));
  const filesToProcess = files.filter(file => {
    return !excludePatterns.some(pattern => {
      return minimatch(file, pattern);
    });
  });

  console.log(`📁 Found ${filesToProcess.length} files to process`);

  let processedCount = 0;
  for (const file of filesToProcess) {
    processedCount += await processFile(file);
  }

  console.log(`\n✨ Complete! Processed ${processedCount} files`);
  console.log('📝 Remember to review the changes and test your application');

  // Create a summary report
  const report = {
    totalFiles: filesToProcess.length,
    processedFiles: processedCount,
    timestamp: new Date().toISOString()
  };

  await fs.writeFile('console-log-replacement-report.json', JSON.stringify(report, null, 2));
  console.log('📊 Report saved to console-log-replacement-report.json');
}

// Run the script
main().catch(console.error);
