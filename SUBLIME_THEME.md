# Sublime Text Theme - GoldenSignals AI

## Overview

The Sublime Text theme brings the familiar, comfortable aesthetics of the popular code editor to the GoldenSignals AI trading platform. This theme offers a softer, more balanced dark mode that's easier on the eyes during extended trading sessions.

## Color Palette

### Primary Colors
- **Background**: `#272822` - Sublime's signature dark gray (softer than pure black)
- **Paper/Cards**: `#3E3D32` - Slightly lighter for elevation
- **Text Primary**: `#F8F8F2` - Warm white for excellent readability
- **Text Secondary**: `#75715E` - Muted gray for secondary information

### Accent Colors
- **Cyan** (Primary): `#66D9EF` - For primary actions and highlights
- **Green** (Success): `#A6E22E` - For positive values and successful operations
- **Pink** (Error): `#F92672` - For negative values and errors
- **Orange** (Warning): `#FD971F` - For warnings and important notices
- **Purple** (Constant): `#AE81FF` - For special highlights
- **Yellow** (String): `#E6DB74` - For additional accents

## Key Differences from Dark Pro

1. **Softer Background**: `#272822` vs `#0A0A0B` - Less harsh on the eyes
2. **Warmer Tones**: The color palette uses warmer, more saturated colors
3. **Better Contrast**: Improved readability with carefully chosen color combinations
4. **Monospace Font**: Uses Consolas/Monaco for a more technical feel
5. **Subtle Borders**: Lower opacity borders for a cleaner look

## Typography

- **Primary Font**: Consolas, Monaco, "Courier New", monospace
- **Fallback Fonts**: System fonts for non-code elements
- **Font Weights**: Lighter weights (400-600) for better readability
- **Letter Spacing**: Optimized for both code and UI elements

## Visual Characteristics

### Cards and Surfaces
- Subtle elevation with minimal shadows
- Soft borders with low opacity
- Smooth hover transitions
- No harsh gradients

### Buttons
- Flat design with subtle hover effects
- Clear active states
- Consistent padding and sizing
- Smooth color transitions

### Charts
- **Candlesticks**: Green (#A6E22E) for up, Pink (#F92672) for down
- **Lines**: Cyan (#66D9EF) for primary data
- **Volume**: Transparent overlays to avoid visual clutter
- **Grid**: Very subtle lines for minimal distraction

### Form Elements
- Dark input backgrounds with subtle borders
- Cyan focus states
- Clear placeholder text
- Smooth transitions

## Usage Examples

### Status Indicators
```css
.sublime-status-success { background: #A6E22E; }
.sublime-status-error { background: #F92672; }
.sublime-status-warning { background: #FD971F; }
```

### Syntax Highlighting Classes
```css
.sublime-keyword { color: #F92672; }
.sublime-string { color: #E6DB74; }
.sublime-function { color: #66D9EF; }
.sublime-variable { color: #FD971F; }
.sublime-comment { color: #75715E; }
.sublime-constant { color: #AE81FF; }
```

## Benefits

1. **Reduced Eye Strain**: Softer background color is easier on the eyes
2. **Better Readability**: Warm white text on dark gray provides excellent contrast
3. **Familiar Feel**: Developers will feel at home with Sublime Text colors
4. **Professional Look**: Clean, minimal design suitable for trading
5. **Consistent Experience**: Colors work well together without clashing

## Accessibility

- **Contrast Ratios**: All text meets WCAG AA standards
- **Color Blindness**: Colors are distinguishable in various color blindness modes
- **Focus Indicators**: Clear cyan outlines for keyboard navigation
- **Hover States**: Subtle but noticeable state changes

## Performance

- Minimal use of gradients and shadows
- Efficient CSS with no heavy effects
- Smooth transitions that don't impact performance
- Optimized for 60fps animations

## Customization

The theme is built with CSS variables and Material-UI theme system, making it easy to:
- Adjust individual colors
- Change font families
- Modify spacing and sizing
- Create variations for different preferences

## Migration from Dark Pro

To switch from Dark Pro to Sublime theme:
1. Import `sublimeTheme` instead of `darkProTheme`
2. Update CSS import to `sublime.css`
3. Chart colors automatically update
4. All components inherit new theme

The Sublime theme provides a more comfortable, familiar environment for extended use while maintaining the professional appearance required for a trading platform. 