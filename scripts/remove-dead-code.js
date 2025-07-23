#!/usr/bin/env node

/**
 * Dead Code Removal Script
 * Safely removes identified dead code from the GoldenSignalsAI project
 */

const fs = require('fs');
const path = require('path');

const projectRoot = path.join(__dirname, '..');
const frontendRoot = path.join(projectRoot, 'frontend');

// Files to remove
const filesToRemove = [
  // Duplicate components at root level
  'frontend/src/components/AISignalProphet.tsx',
  'frontend/src/components/ErrorBoundary.tsx',

  // Disabled test files
  'frontend/src/components/Core/Button/Button.test.tsx.disabled',
  'frontend/src/components/Core/Input/Input.test.tsx.disabled',
  'frontend/src/components/Dashboard/UnifiedDashboard.graph.test.ts.disabled',
  'frontend/src/components/AI/UnifiedAIChat.tdd.test.tsx.disabled',
  'frontend/src/components/MarketInsights/MarketInsights.test.tsx.disabled',
  'frontend/src/components/DebugPanel/DebugPanel.test.tsx.disabled',
  'frontend/src/components/Performance/MetricCard.test.tsx.disabled',
  'frontend/src/components/__tests__/SignalCard.test.tsx.disabled',

  // HybridChart that was never completed
  'frontend/src/components/HybridChart/HybridAITradingChart.tsx',
];

// Files to update
const filesToUpdate = [
  {
    path: 'frontend/src/components/index.ts',
    removals: [
      // Remove all archived chart exports
      'AIPredictionChart',
      'TransformerPredictionChart',
      'MainTradingChart',
      'ChartWrapper',
      'FullChart',
      'CompactChart',
      'MiniChartWrapper',
      'MiniChart',
      'TradingViewDatafeed',
      'CustomChart',
      'EnhancedCustomChart',
      'GoldenSignalsChart',
      'StreamlinedGoldenChart',
      'ProfessionalChart',
      'RealTimeChart',
      'ChartSwitcher'
    ]
  },
  {
    path: 'frontend/src/components/Dashboard/UnifiedDashboard.tsx',
    updates: [
      {
        from: "import { ProfessionalChart } from '../_archived_charts/ProfessionalChart/ProfessionalChart';",
        to: "import { AITradingChart } from '../AIChart/AITradingChart';"
      },
      {
        from: "<ProfessionalChart",
        to: "<AITradingChart"
      }
    ]
  }
];

// Directories to consider removing (after manual review)
const directoriesForReview = [
  'frontend/src/components/_archived_charts',
  'frontend/src/components/HybridChart',
];

console.log('🧹 Starting dead code removal...\n');

// Remove files
console.log('📄 Removing dead files:');
filesToRemove.forEach(file => {
  const filePath = path.join(projectRoot, file);
  if (fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
      console.log(`  ✅ Removed: ${file}`);
    } catch (error) {
      console.log(`  ❌ Error removing ${file}: ${error.message}`);
    }
  } else {
    console.log(`  ⚠️  Not found: ${file}`);
  }
});

// Update files
console.log('\n📝 Updating files:');
filesToUpdate.forEach(({ path: filePath, removals, updates }) => {
  const fullPath = path.join(projectRoot, filePath);

  if (!fs.existsSync(fullPath)) {
    console.log(`  ⚠️  File not found: ${filePath}`);
    return;
  }

  try {
    let content = fs.readFileSync(fullPath, 'utf8');
    let modified = false;

    // Remove exports
    if (removals) {
      removals.forEach(exportName => {
        // Match various export patterns
        const patterns = [
          new RegExp(`export\\s*{[^}]*\\b${exportName}\\b[^}]*}\\s*from\\s*['"][^'"]+['"];?\\n?`, 'g'),
          new RegExp(`export\\s*\\*\\s*as\\s*${exportName}\\s*from\\s*['"][^'"]+['"];?\\n?`, 'g'),
          new RegExp(`export\\s*{\\s*default\\s*as\\s*${exportName}\\s*}\\s*from\\s*['"][^'"]+['"];?\\n?`, 'g'),
        ];

        patterns.forEach(pattern => {
          if (pattern.test(content)) {
            content = content.replace(pattern, '');
            modified = true;
            console.log(`  ✅ Removed export: ${exportName} from ${filePath}`);
          }
        });
      });
    }

    // Apply updates
    if (updates) {
      updates.forEach(({ from, to }) => {
        if (content.includes(from)) {
          content = content.replace(new RegExp(from, 'g'), to);
          modified = true;
          console.log(`  ✅ Updated: ${from} → ${to} in ${filePath}`);
        }
      });
    }

    if (modified) {
      fs.writeFileSync(fullPath, content);
    }
  } catch (error) {
    console.log(`  ❌ Error updating ${filePath}: ${error.message}`);
  }
});

// Report directories for manual review
console.log('\n📁 Directories to review for removal:');
directoriesForReview.forEach(dir => {
  const fullPath = path.join(projectRoot, dir);
  if (fs.existsSync(fullPath)) {
    const files = fs.readdirSync(fullPath);
    console.log(`  • ${dir} (${files.length} files)`);
  }
});

console.log('\n✨ Dead code removal complete!');
console.log('\n⚠️  Important next steps:');
console.log('1. Run "npm run lint" to check for any issues');
console.log('2. Run "npm run typecheck" to ensure type safety');
console.log('3. Test the application thoroughly');
console.log('4. Consider removing the _archived_charts directory if confirmed unused');
console.log('5. Consolidate WebSocket implementations');
