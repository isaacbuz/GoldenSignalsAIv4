#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const componentName = process.argv[2];
const componentType = process.argv[3] || 'component'; // component, page, util

if (!componentName) {
    console.error('Usage: npm run generate:component ComponentName [type]');
    console.error('Types: component (default), page, util');
    process.exit(1);
}

const templates = {
    component: `import React from 'react';
import { Box, Typography } from '@mui/material';
import { useAppSelector } from '../../store/store';
import { selectUserTheme } from '../../store/selectors';

interface ${componentName}Props {
  // Define props here
  className?: string;
  children?: React.ReactNode;
}

export const ${componentName}: React.FC<${componentName}Props> = ({
  className,
  children,
  ...props
}) => {
  const theme = useAppSelector(selectUserTheme);

  return (
    <Box
      data-testid="${componentName.toLowerCase()}"
      className={className}
      sx={{
        // Add styles here
      }}
      {...props}
    >
      <Typography variant="h6">
        ${componentName} Component
      </Typography>
      {children}
    </Box>
  );
};

export default ${componentName};`,

    page: `import React from 'react';
import { Box, Container, Typography } from '@mui/material';
import { useAppSelector } from '../../store/store';
import { selectUserTheme } from '../../store/selectors';

export const ${componentName}Page: React.FC = () => {
  const theme = useAppSelector(selectUserTheme);

  return (
    <Container maxWidth="xl" data-testid="${componentName.toLowerCase()}-page">
      <Box sx={{ py: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          ${componentName}
        </Typography>
        
        {/* Page content goes here */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="body1">
            ${componentName} page content
          </Typography>
        </Box>
      </Box>
    </Container>
  );
};

export default ${componentName}Page;`,

    util: `import React from 'react';

interface ${componentName}Props {
  // Define utility props here
}

export const ${componentName}: React.FC<${componentName}Props> = (props) => {
  // Utility component logic
  return null;
};

// Utility functions
export const use${componentName} = () => {
  // Custom hook logic
  return {};
};

export default ${componentName};`,

    test: `import { render, screen } from '@testing-library/react';
import { ${componentName} } from './${componentName}';
import { TestProviders } from '../../test/TestProviders';

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <${componentName} {...props} />
    </TestProviders>
  );
};

describe('${componentName}', () => {
  it('renders without crashing', () => {
    renderComponent();
    expect(screen.getByTestId('${componentName.toLowerCase()}')).toBeInTheDocument();
  });

  it('displays the component name', () => {
    renderComponent();
    expect(screen.getByText('${componentName} Component')).toBeInTheDocument();
  });

  it('accepts custom className', () => {
    const customClass = 'custom-class';
    renderComponent({ className: customClass });
    const element = screen.getByTestId('${componentName.toLowerCase()}');
    expect(element).toHaveClass(customClass);
  });

  it('renders children when provided', () => {
    const childText = 'Test child content';
    render(
      <TestProviders>
        <${componentName}>
          <div>{childText}</div>
        </${componentName}>
      </TestProviders>
    );
    expect(screen.getByText(childText)).toBeInTheDocument();
  });
});`,

    pageTest: `import { render, screen } from '@testing-library/react';
import { ${componentName}Page } from './${componentName}Page';
import { TestProviders } from '../../test/TestProviders';

describe('${componentName}Page', () => {
  it('renders without crashing', () => {
    render(
      <TestProviders>
        <${componentName}Page />
      </TestProviders>
    );
    expect(screen.getByTestId('${componentName.toLowerCase()}-page')).toBeInTheDocument();
  });

  it('displays the page title', () => {
    render(
      <TestProviders>
        <${componentName}Page />
      </TestProviders>
    );
    expect(screen.getByRole('heading', { name: '${componentName}' })).toBeInTheDocument();
  });
});`,

    story: `import type { Meta, StoryObj } from '@storybook/react';
import { ${componentName} } from './${componentName}';

const meta: Meta<typeof ${componentName}> = {
  title: 'Components/${componentName}',
  component: ${componentName},
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: '${componentName} component description'
      }
    }
  },
  argTypes: {
    className: {
      control: 'text',
      description: 'Custom CSS class name'
    }
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const WithCustomClass: Story = {
  args: {
    className: 'custom-styling',
  },
};

export const WithChildren: Story = {
  args: {
    children: <div>Child content example</div>,
  },
};`,

    pageStory: `import type { Meta, StoryObj } from '@storybook/react';
import { ${componentName}Page } from './${componentName}Page';

const meta: Meta<typeof ${componentName}Page> = {
  title: 'Pages/${componentName}Page',
  component: ${componentName}Page,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: '${componentName} page component'
      }
    }
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};`
};

// Determine paths and file names based on type
let componentDir, testTemplate, storyTemplate, componentTemplate;

switch (componentType) {
    case 'page':
        componentDir = `src/pages/${componentName}`;
        componentTemplate = templates.page;
        testTemplate = templates.pageTest;
        storyTemplate = templates.pageStory;
        break;
    case 'util':
        componentDir = `src/utils/${componentName}`;
        componentTemplate = templates.util;
        testTemplate = templates.test;
        storyTemplate = templates.story;
        break;
    default:
        componentDir = `src/components/${componentName}`;
        componentTemplate = templates.component;
        testTemplate = templates.test;
        storyTemplate = templates.story;
}

// Create directory
fs.mkdirSync(componentDir, { recursive: true });

// Generate files
const files = [
    {
        name: componentType === 'page' ? `${componentName}Page.tsx` : `${componentName}.tsx`,
        content: componentTemplate
    },
    {
        name: componentType === 'page' ? `${componentName}Page.test.tsx` : `${componentName}.test.tsx`,
        content: testTemplate
    },
    {
        name: componentType === 'page' ? `${componentName}Page.stories.tsx` : `${componentName}.stories.tsx`,
        content: storyTemplate
    }
];

files.forEach(({ name, content }) => {
    fs.writeFileSync(path.join(componentDir, name), content);
});

// Update index.ts if it's a component
if (componentType === 'component') {
    const indexPath = 'src/components/index.ts';
    if (fs.existsSync(indexPath)) {
        const indexContent = fs.readFileSync(indexPath, 'utf8');
        const exportLine = `export { ${componentName} } from './${componentName}/${componentName}';`;

        if (!indexContent.includes(exportLine)) {
            fs.appendFileSync(indexPath, `\n${exportLine}`);
        }
    }
}

console.log(`✅ Generated ${componentName} ${componentType}:`);
console.log(`   📁 ${componentDir}/`);
files.forEach(({ name }) => {
    console.log(`   📄 ${name}`);
});

if (componentType === 'component') {
    console.log(`   📄 Updated src/components/index.ts`);
}

console.log('');
console.log('🚀 Next steps:');
console.log('   1. Implement your component logic');
console.log('   2. Add more test cases');
console.log('   3. Customize Storybook stories');
console.log('   4. Update component props and interfaces'); 