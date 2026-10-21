import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { cn } from "@/lib/format";

// ---------------------------------------------------------------- Button
type Variant = "primary" | "secondary" | "ghost" | "danger" | "outline";
type Size = "sm" | "md" | "lg" | "icon";

const variants: Record<Variant, string> = {
  primary: "bg-brand-600 text-white hover:bg-brand-700 shadow-sm disabled:bg-brand-300",
  secondary: "bg-white text-slate-700 ring-1 ring-inset ring-slate-200 hover:bg-slate-50 shadow-sm",
  outline: "bg-transparent text-brand-700 ring-1 ring-inset ring-brand-200 hover:bg-brand-50",
  ghost: "bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-900",
  danger: "bg-rose-600 text-white hover:bg-rose-700 shadow-sm disabled:bg-rose-300",
};
const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-xs gap-1.5",
  md: "h-9 px-3.5 text-sm gap-2",
  lg: "h-11 px-5 text-sm gap-2",
  icon: "h-8 w-8 justify-center",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: ReactNode;
}

export const buttonClass = (variant: Variant = "primary", size: Size = "md", className?: string) =>
  cn(
    "inline-flex items-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70 whitespace-nowrap",
    variants[variant],
    sizes[size],
    className,
  );

/** A react-router link styled as a button (avoids nesting <button> in <a>). */
export function LinkButton({
  to,
  variant,
  size,
  icon,
  className,
  children,
  ...props
}: { to: string; variant?: Variant; size?: Size; icon?: ReactNode; className?: string; children?: ReactNode } & Omit<
  HTMLAttributes<HTMLAnchorElement>,
  "children"
>) {
  return (
    <Link to={to} className={cn(buttonClass(variant, size, className), size !== "icon" && "justify-center")} {...props}>
      {icon}
      {children}
    </Link>
  );
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", size = "md", loading, icon, className, children, disabled, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={buttonClass(variant, size, className)}
      {...props}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  ),
);
Button.displayName = "Button";

// ---------------------------------------------------------------- Form fields
const fieldBase =
  "block w-full rounded-lg border-0 bg-white text-sm text-slate-900 shadow-sm ring-1 ring-inset ring-slate-200 placeholder:text-slate-400 focus:ring-2 focus:ring-inset focus:ring-brand-500 disabled:bg-slate-50 disabled:text-slate-500 aria-[invalid=true]:ring-rose-400";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => <input ref={ref} className={cn(fieldBase, "h-9 px-3", className)} {...props} />,
);
Input.displayName = "Input";

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea ref={ref} className={cn(fieldBase, "px-3 py-2 leading-relaxed", className)} {...props} />
  ),
);
Textarea.displayName = "Textarea";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => (
    <select ref={ref} className={cn(fieldBase, "h-9 pl-3 pr-8", className)} {...props}>
      {children}
    </select>
  ),
);
Select.displayName = "Select";

export function Field({
  label,
  htmlFor,
  hint,
  error,
  required,
  children,
  className,
}: {
  label: string;
  htmlFor: string;
  hint?: ReactNode;
  error?: string;
  required?: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <label htmlFor={htmlFor} className="block text-sm font-medium text-slate-700">
        {label}
        {required && <span className="ml-0.5 text-rose-500" aria-hidden>*</span>}
      </label>
      {children}
      {error ? (
        <p id={`${htmlFor}-error`} className="text-xs text-rose-600" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p id={`${htmlFor}-hint`} className="text-xs text-slate-500">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- Display
type Tone = "neutral" | "brand" | "success" | "warning" | "danger" | "info";
const tones: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-700 ring-slate-200",
  brand: "bg-brand-50 text-brand-700 ring-brand-200",
  success: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  warning: "bg-amber-50 text-amber-800 ring-amber-200",
  danger: "bg-rose-50 text-rose-700 ring-rose-200",
  info: "bg-sky-50 text-sky-700 ring-sky-200",
};

export function Badge({
  tone = "neutral",
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
        tones[tone],
        className,
      )}
      {...props}
    >
      {children}
    </span>
  );
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-xl border border-slate-200 bg-white shadow-card", className)} {...props} />;
}

export function Spinner({ className, label = "Loading" }: { className?: string; label?: string }) {
  return (
    <span role="status" className="inline-flex items-center">
      <Loader2 className={cn("h-4 w-4 animate-spin text-slate-400", className)} aria-hidden />
      <span className="sr-only">{label}</span>
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-slate-200/70", className)} aria-hidden />;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon: ReactNode;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center px-6 py-14 text-center", className)}>
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-50 text-brand-600 ring-1 ring-brand-100">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      {description && <p className="mt-1 max-w-sm text-sm text-slate-500">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Alert({
  tone = "danger",
  title,
  children,
  action,
}: {
  tone?: "danger" | "warning" | "info";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const styles = {
    danger: "border-rose-200 bg-rose-50 text-rose-800",
    warning: "border-amber-200 bg-amber-50 text-amber-900",
    info: "border-sky-200 bg-sky-50 text-sky-900",
  }[tone];
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={cn("rounded-lg border px-4 py-3 text-sm", styles)}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-medium">{title}</p>
          {children && <div className="mt-1 opacity-90">{children}</div>}
        </div>
        {action}
      </div>
    </div>
  );
}
