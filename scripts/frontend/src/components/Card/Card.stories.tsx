/**
 * Card Storybook Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { Card } from './Card';

const meta = {
  title: 'Components/Card',
  component: Card,
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
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Default story showing the component in its default state
 */
export const Default: Story = {
  args: {
    children: 'Default Card',
  },
};

/**
 * Example with custom styling
 */
export const WithCustomStyle: Story = {
  args: {
    children: 'Styled Card',
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
        <h3>Card with Complex Content</h3>
        <p>This demonstrates how the component handles complex children.</p>
        <button>Interactive Button</button>
      </div>
    ),
  },
};

// TODO: Add more stories demonstrating different states and props
