# Fix Metrics Display and Redesign Dashboard with Shadcn Design Schema

## Problems Identified

1. **Metrics not visible** - CSS may be hiding content or HTML rendering issues with metric cards
2. **UI needs shadcn design schema** - Apply shadcn/ui design principles: clean, minimal, sharp design with subtle shadows
3. **Statistical models need verification** - Ensure calculations match documented thresholds
4. **User-friendliness** - Improve clarity and accessibility
5. **API status checking** - Add API configuration verification and working status indicators
6. **Result display** - Verify results are shown properly and use professional table layouts

## Implementation Plan

### 1. Add API Configuration Check and Status Monitoring

**New Features** (`frontend/dashboard.py`):
- **API Status Indicator**:
  - Check API health endpoint (`/health`) on dashboard load
  - Display connection status with visual indicator (green/red dot)
  - Show API URL and last check time
  - Auto-refresh API status every 30 seconds
  
- **API Configuration Panel**:
  - Display API base URL (from environment/config)
  - Show connection latency
  - Test all endpoints: `/health`, `/api/drift/metrics`, `/api/drift/alerts`
  - Display any connection errors clearly
  
- **Status Checks**:
  - Verify API is reachable
  - Check response times
  - Validate JSON response format
  - Show detailed error messages if API fails

**Implementation**:
- Add `check_api_status()` function
- Create API status widget in sidebar or header
- Add connection test button
- Display error messages if API is unreachable

### 2. Verify Results Display Properly

**Checks Needed**:
- Verify metric cards render correctly
- Check API response format matches expected structure
- Ensure all values display (not None or missing)
- Validate data types and formatting
- Test with empty/null responses gracefully

**Fix Issues**:
- Handle missing/None values in metrics
- Add proper error handling for malformed API responses
- Display "No data" states gracefully
- Add loading states while fetching
- Show timestamps for last successful fetch

### 3. Build Dashboard with Professional Tables

**Replace/Enhance Current Display with Tables**:

- **Metrics Table**:
  - Display metrics in clean table format
  - Columns: Metric Name | Current Value | Threshold | Status | Trend
  - Use shadcn-styled table with proper spacing
  - Sortable columns
  - Color-coded status indicators
  
- **Alerts Table**:
  - Professional alerts table with columns:
    - Timestamp | Alert Type | Metric | Value | Threshold | Severity | Status | Actions
  - Sortable and filterable
  - Inline actions (dismiss, view details)
  - Group by severity
  
- **Drift History Table**:
  - Time-series data in table format
  - Columns: Timestamp | PSI | KS p-value | JS Divergence | Sample Size | Status
  - Pagination for large datasets
  - Export functionality
  
- **Configuration Versions Table**:
  - Table showing all config versions
  - Columns: Version ID | Timestamp | Model | Thresholds | Performance | Status | Actions
  - Show current version indicator
  - Rollback actions in table

**Table Styling (Shadcn Schema)**:
- Clean borders: `1px solid hsl(0 0% 14.9%)`
- Header: `hsl(0 0% 14.9%)` background
- Alternating row colors (subtle)
- Hover effects on rows
- Sharp corners (6px border radius)
- Professional typography

### 4. Fix Metrics Visibility Issues (`frontend/dashboard.py`)

**Issue**: Metric cards may not be rendering due to:
- CSS variables not being properly applied
- HTML structure issues in `create_metric_card()`
- Missing color definitions
- Potential z-index or positioning issues

**Fix**:
- Simplify metric card HTML structure
- Use explicit color values (shadcn color palette) instead of CSS variables
- Add explicit display properties for all elements
- Ensure proper visibility and contrast

### 5. Apply Shadcn Design Schema to CSS

**Shadcn Design Principles** (from shadcn/ui):
- **Color Palette**: Use HSL-based colors
  - Background: `hsl(0 0% 3.9%)` (very dark)
  - Foreground: `hsl(0 0% 98%)` (almost white)
  - Card: `hsl(0 0% 3.9%)` with border `hsl(0 0% 14.9%)`
  - Muted: `hsl(0 0% 14.9%)`
  - Accent borders: `hsl(0 0% 14.9%)`
- **Border Radius**: 
  - Cards: `6px`
  - Buttons: `4px`
  - Tables: `6px` (outer corners only)
- **Shadows**: Subtle, layered shadows (no heavy gradients)
- **Typography**: System font stack `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- **Spacing**: 4px base unit (4px, 8px, 12px, 16px, 24px, 32px)
- **Borders**: 1px solid, subtle colors
- **Tables**: Clean borders, alternating rows, hover effects

**CSS Changes** (`frontend/dashboard.py`, lines 21-210):
- Replace gradient backgrounds with flat shadcn colors
- Use shadcn border radius values (6px for cards, tables)
- Apply shadcn shadow system (subtle box-shadows)
- Use shadcn color palette (HSL values)
- Clean typography with system fonts
- Consistent spacing scale
- Add table-specific CSS styling

### 6. Verify Statistical Models

**Check PSI calculation** (`backend/utils/drift_statistics.py`, lines 8-49):
- PSI formula: `sum((actual - expected) * log(actual/expected))`
- Thresholds: <0.1 normal, 0.1-0.25 warning, >0.25 critical
- Verify binning logic and edge cases

**Check KS Test** (`backend/utils/drift_statistics.py`, lines 52-71):
- Using scipy.stats.ks_2samp correctly
- Lower p-value = more drift (correct interpretation)
- Thresholds: >0.05 normal, 0.01-0.05 warning, <0.01 critical

**Check JS Divergence** (`backend/utils/drift_statistics.py`, lines 74-129):
- Using scipy jensenshannon correctly (returns 0-1)
- Thresholds: <0.1 normal, 0.1-0.2 warning, 0.2-0.3 critical, >0.3 emergency

**Verify Service Logic** (`backend/services/drift_detection_service.py`):
- `compute_drift_metrics()` returns correct structure
- Threshold checks use correct comparisons
- Default values handled properly

### 7. Improve User-Friendliness

**Metric Labels**:
- Keep current user-friendly labels ("Input Shift Detection", etc.)
- Add clear threshold indicators
- Show "last updated" timestamp

**Visual Indicators**:
- Use shadcn-style subtle left border accent for severity (4px width)
- Progress bars with shadcn styling (subtle, clean)
- Status badges with shadcn color scheme

**Layout**:
- Better spacing following shadcn scale (16px, 24px gaps)
- Clear section headers
- Improved empty states
- Professional table layouts for data display

## Files to Modify

1. **[frontend/dashboard.py](frontend/dashboard.py)**
   - Lines 21-210: Apply shadcn design schema to CSS (add table styles)
   - Lines 208-260: Add API status checking functions
   - Lines 339-375: Metric card function with shadcn styling
   - Lines 410-416: Add API status widget in sidebar/header
   - Lines 500-600: Replace metric cards with professional tables
   - Lines 568-650: Convert alerts to table format
   - Lines 650-750: Convert history to table format
   - Lines 750-850: Convert config versions to table format

2. **[backend/utils/drift_statistics.py](backend/utils/drift_statistics.py)**
   - Review and verify all statistical calculations
   - Add comments clarifying threshold meanings

3. **[backend/services/drift_detection_service.py](backend/services/drift_detection_service.py)**
   - Verify `compute_drift_metrics()` return format
   - Check threshold comparison logic

## Shadcn Design Schema Reference

**Colors** (using HSL):
- `background`: `hsl(0 0% 3.9%)`
- `foreground`: `hsl(0 0% 98%)`
- `card`: `hsl(0 0% 3.9%)`
- `card-foreground`: `hsl(0 0% 98%)`
- `border`: `hsl(0 0% 14.9%)`
- `muted`: `hsl(0 0% 14.9%)`
- `muted-foreground`: `hsl(0 0% 63.9%)`

**Severity Colors** (applied to shadcn base):
- Normal: Green (`hsl(142 76% 36%)`)
- Warning: Orange (`hsl(38 92% 50%)`)
- Critical: Red (`hsl(0 84% 60%)`)
- Emergency: Dark Red (`hsl(0 72% 51%)`)

**Shadows**:
- Card: `0 2px 8px rgba(0, 0, 0, 0.15)`
- Hover: `0 4px 12px rgba(0, 0, 0, 0.2)`
- Table row hover: `background: hsl(0 0% 14.9%)`

**Table Styling**:
- Border: `1px solid hsl(0 0% 14.9%)`
- Header background: `hsl(0 0% 14.9%)`
- Row hover: `hsl(0 0% 14.9%)`
- Alternating rows: Subtle background difference