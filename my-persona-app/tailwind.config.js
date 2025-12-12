/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                'myeongjo': ['Nanum Myeongjo', 'serif'],
                'square': ['Nanum Square', 'sans-serif'],
            },
        },
    },
    plugins: [],
}