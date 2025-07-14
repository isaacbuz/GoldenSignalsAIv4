import type { Meta, StoryObj } from '@storybook/react';
import { ProphetOrb } from './ProphetOrb';

const meta: Meta<typeof ProphetOrb> = {
  title: 'Components/ProphetOrb',
  component: ProphetOrb,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'ProphetOrb component description'
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