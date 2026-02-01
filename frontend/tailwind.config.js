/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                mono: ['JetBrains Mono', 'SF Mono', 'Consolas', 'monospace'],
            },
            colors: {
                // YC-inspired color palette
                brand: {
                    DEFAULT: '#f26625',
                    hover: '#e55a1b',
                    light: '#fff7f3',
                    muted: 'rgba(242, 102, 37, 0.08)',
                },
                surface: {
                    DEFAULT: '#ffffff',
                    secondary: '#fafafa',
                    tertiary: '#f5f5f5',
                },
                border: {
                    DEFAULT: '#e5e5e5',
                    light: '#f0f0f0',
                    dark: '#d4d4d4',
                },
                text: {
                    primary: '#171717',
                    secondary: '#525252',
                    muted: '#a3a3a3',
                },
                success: '#16a34a',
                danger: '#dc2626',
                warning: '#ca8a04',
            },
            boxShadow: {
                'sm': '0 1px 2px rgba(0, 0, 0, 0.04)',
                'DEFAULT': '0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)',
                'md': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
                'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.03)',
                'card': '0 0 0 1px rgba(0,0,0,0.03), 0 2px 4px rgba(0,0,0,0.05)',
            },
            borderRadius: {
                'xl': '12px',
                '2xl': '16px',
            },
        },
    },
    plugins: [],
}
