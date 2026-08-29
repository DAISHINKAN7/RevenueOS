import { EmptyState } from "@/components/ui/primitives";

export default function NotFound() {
  return <EmptyState title="Page not found" hint="The route you requested does not exist." />;
}