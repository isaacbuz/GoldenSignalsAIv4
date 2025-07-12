import type { Meta, StoryObj } from '@storybook/react';
import { SignalCard } from './SignalCard';

const meta: Meta<typeof SignalCard> = {
  title: 'Components/SignalCard',
  component: SignalCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'SignalCard component description'
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