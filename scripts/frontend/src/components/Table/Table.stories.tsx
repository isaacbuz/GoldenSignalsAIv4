/**
 * Table Storybook Stories
 */

import type { Meta, StoryObj } from '@storybook/react';
import { Table } from './Table';

const meta = {
  title: 'Components/Table',
  component: Table,
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
} satisfies Meta<typeof Table>;

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Default story showing the component in its default state
 */
export const Default: Story = {
  args: {
    children: 'Default Table',
  },
};

/**
 * Example with custom styling
 */
export const WithCustomStyle: Story = {
  args: {
    children: 'Styled Table',
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
        <h3>Table with Complex Content</h3>
        <p>This demonstrates how the component handles complex children.</p>
        <button>Interactive Button</button>
      </div>
    ),
  },
};

// TODO: Add more stories demonstrating different states and props
