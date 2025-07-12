import type { Meta, StoryObj } from '@storybook/react';
import { AdvancedChart } from './AdvancedChart';

const meta: Meta<typeof AdvancedChart> = {
  title: 'Components/AdvancedChart',
  component: AdvancedChart,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'AdvancedChart component description'
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