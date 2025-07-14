import type { Meta, StoryObj } from '@storybook/react';
import { AnalysisLegend } from './AnalysisLegend';

const meta: Meta<typeof AnalysisLegend> = {
  title: 'Components/AnalysisLegend',
  component: AnalysisLegend,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'AnalysisLegend component description'
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