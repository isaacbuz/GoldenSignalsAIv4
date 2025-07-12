/**
 * Error Storybook Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { Error } from './Error';

const meta = {
  title: 'Components/Error',
  component: Error,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'TODO: Add component description for Storybook docs',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
    'data-testid': {
      control: 'text',
      description: 'Test ID for testing',
    },
  },
} satisfies Meta<typeof Error>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Default story showing the component in its default state
 */
export const Default: Story = {
  args: {
    children: 'Default Error',
  },
};

/**
 * Example with custom styling
 */
export const WithCustomStyle: Story = {
  args: {
    children: 'Styled Error',
    className: 'custom-style',
  },
};

/**
 * Example with complex children
 */
export const WithComplexChildren: Story = {
  args: {
    children: (
      <div>
        <h3>Error with Complex Content</h3>
        <p>This demonstrates how the component handles complex children.</p>
        <button>Interactive Button</button>
      </div>
    ),
  },
};

// TODO: Add more stories demonstrating different states and props
