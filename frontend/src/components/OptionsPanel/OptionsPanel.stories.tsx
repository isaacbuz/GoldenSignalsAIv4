import type { Meta, StoryObj } from '@storybook/react';
import { OptionsPanel } from './OptionsPanel';

const meta: Meta<typeof OptionsPanel> = {
  title: 'Components/OptionsPanel',
  component: OptionsPanel,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'OptionsPanel component description'
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