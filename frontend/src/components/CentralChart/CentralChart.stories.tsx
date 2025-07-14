import type { Meta, StoryObj } from '@storybook/react';
import { CentralChart } from './CentralChart';

const meta: Meta<typeof CentralChart> = {
  title: 'Components/CentralChart',
  component: CentralChart,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'CentralChart component description'
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
};