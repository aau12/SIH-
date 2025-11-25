# GNSS Forecasting Frontend

## 🚀 Overview

This is a modern React-based frontend for the GNSS Forecasting System, built with:
- **React 18.3** - Modern UI framework
- **Vite 6.3** - Fast build tool and dev server
- **TypeScript** - Type-safe development
- **Radix UI** - Accessible component primitives
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization
- **Lucide React** - Beautiful icons

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/      # React components (71 items)
│   ├── styles/          # Custom styles
│   ├── guidelines/      # Design guidelines
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   ├── index.css        # Global styles
│   └── Attributions.md  # Credits and attributions
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
├── vite.config.ts       # Vite configuration
└── README.md            # Project documentation
```

## 🔧 Installation

### Prerequisites
- Node.js (v18 or higher recommended)
- npm or yarn package manager

### Setup Steps

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173`

4. **Build for production**
   ```bash
   npm run build
   ```

## 📦 Key Dependencies

### UI Components
- `@radix-ui/*` - Accessible, unstyled UI primitives
- `lucide-react` - Icon library
- `next-themes` - Dark mode support
- `sonner` - Toast notifications
- `vaul` - Drawer component

### Data Visualization
- `recharts` - Charting library for React

### Forms & Interactions
- `react-hook-form` - Form validation
- `react-day-picker` - Date picker
- `cmdk` - Command palette
- `embla-carousel-react` - Carousel component

### Styling
- `tailwind-merge` - Merge Tailwind classes
- `class-variance-authority` - Variant management
- `clsx` - Conditional class names

## 🎨 Features

### Modern UI/UX
- ✅ Responsive design
- ✅ Dark mode support
- ✅ Accessible components
- ✅ Smooth animations
- ✅ Interactive charts

### Components
- ✅ 71+ reusable components
- ✅ Form controls (inputs, selects, checkboxes, etc.)
- ✅ Navigation (menus, tabs, breadcrumbs)
- ✅ Overlays (dialogs, popovers, tooltips)
- ✅ Data display (tables, cards, charts)
- ✅ Feedback (alerts, toasts, progress bars)

## 🔗 Integration with Backend

The frontend is designed to integrate with the Python backend APIs:

### API Endpoints (Expected)
- `/api/predictions` - Get GNSS predictions
- `/api/models` - Model information
- `/api/realtime` - Real-time data updates
- `/api/evaluation` - Model evaluation metrics

### Configuration
Update API endpoints in your configuration file or environment variables.

## 🛠️ Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

### Development Server
- **URL**: http://localhost:5173
- **Hot Module Replacement**: Enabled
- **Fast Refresh**: Enabled

## 📝 Migration Notes

### Previous Dashboard (Streamlit)
The old Streamlit-based dashboard has been replaced with this React frontend for:
- Better performance
- Modern UI/UX
- Enhanced interactivity
- Easier customization
- Better mobile support

### Key Differences
| Feature | Old (Streamlit) | New (React) |
|---------|----------------|-------------|
| Framework | Python/Streamlit | React/TypeScript |
| Styling | Custom CSS | Tailwind CSS |
| Components | Streamlit widgets | Radix UI + Custom |
| Charts | Plotly | Recharts |
| State Management | Session state | React hooks |
| Deployment | Streamlit Cloud | Static hosting |

## 🚀 Deployment

### Static Hosting
Build the app and deploy to:
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy`
- **GitHub Pages**: Build and push to gh-pages branch
- **AWS S3**: Upload build folder to S3 bucket

### Build Output
The `npm run build` command creates a `dist/` folder with optimized static files.

## 🔧 Configuration

### Vite Config
Edit `vite.config.ts` to customize:
- Build options
- Dev server settings
- Plugin configuration
- Path aliases

### Environment Variables
Create a `.env` file for environment-specific settings:
```env
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=GNSS Forecasting
```

## 📚 Resources

- [React Documentation](https://react.dev)
- [Vite Documentation](https://vitejs.dev)
- [Radix UI](https://www.radix-ui.com)
- [Tailwind CSS](https://tailwindcss.com)
- [Recharts](https://recharts.org)

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use a different port
npm run dev -- --port 3000
```

### Module Not Found
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Build Errors
```bash
# Clear Vite cache
rm -rf node_modules/.vite
npm run dev
```

## 🎯 Next Steps

1. **Install dependencies**: `cd frontend && npm install`
2. **Start dev server**: `npm run dev`
3. **Connect to backend**: Configure API endpoints
4. **Customize**: Modify components and styles as needed
5. **Deploy**: Build and deploy to your hosting platform

## 📧 Support

For issues or questions:
- Check the README in the frontend directory
- Review component documentation
- Check Vite and React documentation

---

**Happy Coding! 🛰️**
