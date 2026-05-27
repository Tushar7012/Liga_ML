import { createTheme, type ThemeOptions } from '@mui/material/styles';

// ── Shared tokens ────────────────────────────────────────────────
const sharedTypography: ThemeOptions['typography'] = {
  fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
  fontSize: 15,
  button: {
    fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
    textTransform: 'none' as const,
    fontWeight: 600,
  },
};

const sharedComponents: ThemeOptions['components'] = {
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: '10px',
        fontWeight: 600,
        transition: 'transform 0.06s ease, background 0.12s ease, box-shadow 0.12s ease',
        '&:hover': { transform: 'translateY(-1px)' },
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: { backgroundImage: 'none' },
    },
  },
};

const sharedShape: ThemeOptions['shape'] = { borderRadius: 12 };

// ── Ligaments.ai light palette ───────────────────────────────────
const darkVars = {
  '--bg': '#FFFFFF',
  '--panel': '#F7FCF8',
  '--surface': '#FFFFFF',
  '--text': '#101914',
  '--muted-text': '#607568',
  '--accent-yellow': '#00D084',
  '--accent-yellow-weak': 'rgba(0,208,132,0.1)',
  '--accent-green': '#00B875',
  '--accent-red': '#E05A4F',
  '--shadow-1': '0 12px 30px rgba(0, 96, 61, 0.1)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(0,208,132,0.16)',
  '--border': 'rgba(0, 128, 82, 0.1)',
  '--border-hover': 'rgba(0, 128, 82, 0.22)',
  '--code-bg': 'rgba(0,128,82,0.06)',
  '--tool-bg': 'rgba(0,208,132,0.06)',
  '--tool-border': 'rgba(0,128,82,0.12)',
  '--hover-bg': 'rgba(0,208,132,0.08)',
  '--composer-bg': '#FFFFFF',
  '--msg-gradient': 'linear-gradient(180deg, rgba(0,208,132,0.04), transparent)',
  '--body-gradient': 'linear-gradient(180deg, #FFFFFF 0%, #F3FBF6 100%)',
  '--scrollbar-thumb': '#B7DBC8',
  '--success-icon': '#00B875',
  '--error-icon': '#F87171',
  '--clickable-text': 'rgba(16,25,20,0.92)',
  '--clickable-underline': 'rgba(0,128,82,0.26)',
  '--code-panel-bg': '#F5FBF7',
  '--tab-active-bg': 'rgba(0,208,132,0.12)',
  '--tab-active-border': 'rgba(0,128,82,0.18)',
  '--tab-hover-bg': 'rgba(0,208,132,0.08)',
  '--tab-close-hover': 'rgba(0,128,82,0.1)',
  '--plan-bg': 'rgba(0,208,132,0.06)',
} as const;

// ── Light palette ────────────────────────────────────────────────
const lightVars = {
  '--bg': '#FFFFFF',
  '--panel': '#F7FCF8',
  '--surface': '#FFFFFF',
  '--text': '#101914',
  '--muted-text': '#607568',
  '--accent-yellow': '#00D084',
  '--accent-yellow-weak': 'rgba(0,208,132,0.1)',
  '--accent-green': '#00B875',
  '--accent-red': '#DC2626',
  '--shadow-1': '0 12px 30px rgba(0, 96, 61, 0.1)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(0,208,132,0.16)',
  '--border': 'rgba(0, 128, 82, 0.1)',
  '--border-hover': 'rgba(0, 128, 82, 0.22)',
  '--code-bg': 'rgba(0,128,82,0.06)',
  '--tool-bg': 'rgba(0,208,132,0.06)',
  '--tool-border': 'rgba(0,128,82,0.12)',
  '--hover-bg': 'rgba(0,208,132,0.08)',
  '--composer-bg': '#FFFFFF',
  '--msg-gradient': 'linear-gradient(180deg, rgba(0,208,132,0.04), transparent)',
  '--body-gradient': 'linear-gradient(180deg, #FFFFFF 0%, #F3FBF6 100%)',
  '--scrollbar-thumb': '#B7DBC8',
  '--success-icon': '#00B875',
  '--error-icon': '#DC2626',
  '--clickable-text': 'rgba(16,25,20,0.92)',
  '--clickable-underline': 'rgba(0,128,82,0.26)',
  '--code-panel-bg': '#F5FBF7',
  '--tab-active-bg': 'rgba(0,208,132,0.12)',
  '--tab-active-border': 'rgba(0,128,82,0.18)',
  '--tab-hover-bg': 'rgba(0,208,132,0.08)',
  '--tab-close-hover': 'rgba(0,128,82,0.1)',
  '--plan-bg': 'rgba(0,208,132,0.06)',
} as const;

// ── Shared CSS baseline (scrollbar, code, brand-logo) ────────────
function makeCssBaseline(vars: Record<string, string>) {
  return {
    styleOverrides: {
      ':root': vars,
      body: {
        background: 'var(--body-gradient)',
        color: 'var(--text)',
        scrollbarWidth: 'thin' as const,
        '&::-webkit-scrollbar': { width: '8px', height: '8px' },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: 'var(--scrollbar-thumb)',
          borderRadius: '2px',
        },
        '&::-webkit-scrollbar-track': { backgroundColor: 'transparent' },
      },
      'code, pre': {
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace',
      },
      '.brand-logo': {
        position: 'relative' as const,
        padding: '6px',
        borderRadius: '8px',
        '&::after': {
          content: '""',
          position: 'absolute' as const,
          inset: '-6px',
          borderRadius: '10px',
          background: 'var(--accent-yellow-weak)',
          zIndex: -1,
          pointerEvents: 'none' as const,
        },
      },
    },
  };
}

function makeDrawer() {
  return {
    styleOverrides: {
      paper: {
        backgroundColor: 'var(--panel)',
        borderRight: '1px solid var(--border)',
      },
    },
  };
}

function makeTextField() {
  return {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 'var(--radius-md)',
          '& fieldset': { borderColor: 'var(--border)' },
          '&:hover fieldset': { borderColor: 'var(--border-hover)' },
          '&.Mui-focused fieldset': {
            borderColor: 'var(--accent-yellow)',
            borderWidth: '1px',
            boxShadow: 'var(--focus)',
          },
        },
      },
    },
  };
}

// ── Theme builders ───────────────────────────────────────────────
export const darkTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#00D084', light: '#36E0A2', dark: '#00A96B', contrastText: '#062B1D' },
    secondary: { main: '#00A96B' },
    background: { default: '#FFFFFF', paper: '#F7FCF8' },
    text: { primary: '#101914', secondary: '#607568' },
    divider: 'rgba(0,128,82,0.1)',
    success: { main: '#00B875' },
    error: { main: '#E05A4F' },
    warning: { main: '#00D084' },
    info: { main: '#15803D' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(darkVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#00D084', light: '#36E0A2', dark: '#00A96B', contrastText: '#062B1D' },
    secondary: { main: '#00A96B' },
    background: { default: '#FFFFFF', paper: '#F7FCF8' },
    text: { primary: '#101914', secondary: '#607568' },
    divider: 'rgba(0,128,82,0.1)',
    success: { main: '#00B875' },
    error: { main: '#DC2626' },
    warning: { main: '#00D084' },
    info: { main: '#15803D' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(lightVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

// Keep default export for backwards compat.
export default lightTheme;
