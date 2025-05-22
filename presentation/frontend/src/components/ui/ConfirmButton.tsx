import React, { useState } from "react";
import { Dialog } from "@headlessui/react";

interface Props {
  title: string;
  description: string;
  onConfirm: () => void;
  children: React.ReactNode;
  className?: string;
}

const ConfirmButton: React.FC<Props> = ({ title, description, onConfirm, children, className }) => {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button onClick={() => setOpen(true)} className={className}>
        {children}
      </button>

      <Dialog open={open} onClose={() => setOpen(false)} className="relative z-50">
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" aria-hidden="true" />
        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Dialog.Panel className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-sm text-white shadow-xl">
            <Dialog.Title className="text-lg font-bold mb-2">{title}</Dialog.Title>
            <p className="text-sm text-gray-300 mb-4">{description}</p>
            <div className="flex justify-end space-x-2">
              <button onClick={() => setOpen(false)} className="px-3 py-1 bg-gray-600 rounded">
                Cancel
              </button>
              <button
                onClick={() => {
                  setOpen(false);
                  onConfirm();
                }}
                className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-black font-semibold"
              >
                Confirm
              </button>
            </div>
          </Dialog.Panel>
        </div>
      </Dialog>
    </>
  );
};

export default ConfirmButton;
