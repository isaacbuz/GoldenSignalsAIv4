import type { Meta, StoryObj } from '@storybook/react';
import { RealTimeFeed } from './RealTimeFeed';

const meta: Meta<typeof RealTimeFeed> = {
  title: 'Components/RealTimeFeed',
  component: RealTimeFeed,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'RealTimeFeed component description'
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