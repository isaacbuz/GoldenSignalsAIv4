# Next.js Frontend Migration Plan

This directory contains the new Next.js frontend for GoldenSignalsAI, scaffolded with TypeScript, Tailwind CSS, ESLint, and App Router for a modern, beautiful, and scalable UI.

## Migration Steps
1. **Component Audit:**
   - Review all React components in `presentation/frontend/src/`.
   - Identify core UI elements: dashboards, admin panels, agent controls, charts, onboarding, login, etc.

2. **Component Migration:**
   - Migrate each JS component to the new Next.js app as a `.tsx` file.
   - Refactor to use Tailwind CSS for styling (replace legacy CSS imports).
   - Modularize: group related components in subfolders (e.g., `admin/`, `auth/`, `dashboard/`).

3. **App Structure:**
   - Use the `app/` directory for routing and layout.
   - Place global providers (theme, state, agentic context) in `layout.tsx`.
   - Add pages for dashboard, login, onboarding, analytics, etc.

4. **Backend Integration:**
   - Connect to backend APIs for agentic model results, analytics, and admin controls.
   - Show which model/provider generated each result; allow user selection or comparison.

5. **UI/UX Enhancements:**
   - Add loading, error, and empty states.
   - Use LLMs for explanations, onboarding, and tooltips.
   - Ensure all components are accessible and responsive.

6. **Testing & Documentation:**
   - Add tests for key components.
   - Document each module and component for maintainability.

---

**This migration will result in a clean, beautiful, and agentic UI, fully leveraging the power of Next.js and Tailwind CSS.**
