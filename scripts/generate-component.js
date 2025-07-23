#!/usr/bin/env node

/**
 * Component Generator Script
 * Creates a new React component with all required files:
 * - Component file with logging
 * - Test file with data-testids

 * - CSS module
 */

const fs = require('fs');
const path = require('path');

// Get component name from command line
const componentName = process.argv[2];

if (!componentName) {
    console.error('Please provide a component name');
    console.error('Usage: node scripts/generate-component.js ComponentName');
    process.exit(1);
}

// Validate component name
if (!/^[A-Z][A-Za-z0-9]*$/.test(componentName)) {
    console.error('Component name must start with uppercase letter and contain only letters and numbers');
    process.exit(1);
}

// Convert to various case formats
const kebabCase = componentName
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .toLowerCase();

const camelCase = componentName.charAt(0).toLowerCase() + componentName.slice(1);

// Define paths
const componentDir = path.join('frontend', 'src', 'components', componentName);
const indexPath = path.join(componentDir, 'index.tsx');
const componentPath = path.join(componentDir, `${componentName}.tsx`);
const testPath = path.join(componentDir, `${componentName}.test.tsx`);

const stylePath = path.join(componentDir, `${componentName}.module.css`);

// Component template
const componentTemplate = `/**
 * ${componentName} Component
 *
 * TODO: Add component description
 */

import React from 'react';
import { useComponentLogger } from '@/services/logging/logger';
import { testIds } from '@/utils/test-utils';
import styles from './${componentName}.module.css';

export interface ${componentName}Props {
  /**
   * Component ID for testing
   */
  'data-testid'?: string;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Children elements
   */
  children?: React.ReactNode;
  // TODO: Add more props
}

export const ${componentName}: React.FC<${componentName}Props> = ({
  'data-testid': testId = '${kebabCase}',
  className,
  children,
  ...props
}) => {
  const logger = useComponentLogger('${componentName}');

  // Log render
  logger.debug('Rendering', { props });

  return (
    <div
      className={\`\${styles.container} \${className || ''}\`}
      data-testid={testId}
      {...props}
    >
      {children}
    </div>
  );
};

${componentName}.displayName = '${componentName}';
`;

// Index file template
const indexTemplate = `export { ${componentName} } from './${componentName}';
export type { ${componentName}Props } from './${componentName}';
`;

// Test file template
const testTemplate = `/**
 * ${componentName} Component Tests
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ${componentName} } from './${componentName}';
import { testIds } from '@/utils/test-utils';

describe('${componentName}', () => {
  it('renders without crashing', () => {
    render(<${componentName} />);
    expect(screen.getByTestId('${kebabCase}')).toBeInTheDocument();
  });

  it('accepts custom data-testid', () => {
    const customTestId = 'custom-test-id';
    render(<${componentName} data-testid={customTestId} />);
    expect(screen.getByTestId(customTestId)).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    render(<${componentName} className={customClass} />);
    const element = screen.getByTestId('${kebabCase}');
    expect(element).toHaveClass(customClass);
  });

  it('renders children', () => {
    const childText = 'Test child content';
    render(
      <${componentName}>
        <span>{childText}</span>
      </${componentName}>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });

  // TODO: Add more specific tests for your component
});
`;



// CSS Module template
const cssTemplate = `/**
 * ${componentName} Styles
 */

.container {
  /* Base container styles */
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;

  /* TODO: Add component-specific styles */
}

/* Responsive styles */
@media (max-width: 768px) {
  .container {
    /* Mobile styles */
  }
}

/* Dark theme support */
[data-theme="dark"] .container {
  /* Dark theme overrides */
}

/* Animation classes */
.container {
  transition: all 0.2s ease-in-out;
}

.container:hover {
  /* Hover state */
}

/* State variations */
.container.active {
  /* Active state */
}

.container.disabled {
  /* Disabled state */
  opacity: 0.6;
  pointer-events: none;
}
`;

// Create component directory
if (!fs.existsSync(componentDir)) {
    fs.mkdirSync(componentDir, { recursive: true });
} else {
    console.error(`Component directory already exists: ${componentDir}`);
    process.exit(1);
}

// Write all files
try {
    fs.writeFileSync(indexPath, indexTemplate);
    fs.writeFileSync(componentPath, componentTemplate);
    fs.writeFileSync(testPath, testTemplate);

    fs.writeFileSync(stylePath, cssTemplate);

    console.log(`✅ Component "${componentName}" generated successfully!`);
    console.log('\nCreated files:');
    console.log(`  📄 ${indexPath}`);
    console.log(`  🧩 ${componentPath}`);
    console.log(`  🧪 ${testPath}`);

    console.log(`  🎨 ${stylePath}`);
    console.log('\nNext steps:');
    console.log(`  1. Update the component logic in ${componentPath}`);
    console.log(`  2. Add specific tests in ${testPath}`);
    console.log(`  3. Style the component in ${stylePath}`);

} catch (error) {
    console.error('Error creating component files:', error);
    process.exit(1);
}
