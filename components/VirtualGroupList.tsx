import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from "react"
import type { ReactNode } from "react"
import { VariableSizeList } from "react-window"
import type { ListChildComponentProps } from "react-window"

interface RowData {
  render: (index: number) => ReactNode
  measure: (index: number, height: number) => void
}

function MeasuredRow({ index, style, data }: ListChildComponentProps<RowData>) {
  const content = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const element = content.current
    if (!element) return
    const measure = () => {
      const height = element.getBoundingClientRect().height
      if (height > 0) data.measure(index, Math.ceil(height))
    }
    measure()
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [index, data])
  return (
    <div style={style}>
      <div ref={content} style={{ display: "flow-root" }}>
        {data.render(index)}
      </div>
    </div>
  )
}

/** Variable-height rows retain wrapped photo layouts while unmounting distant groups. */
export function VirtualGroupList({
  ids,
  render
}: {
  ids: string[]
  render: (index: number) => ReactNode
}) {
  const list = useRef<VariableSizeList<RowData>>(null)
  const container = useRef<HTMLDivElement>(null)
  const heights = useRef(new Map<number, number>())
  const [height, setHeight] = useState(() =>
    typeof window === "undefined"
      ? 600
      : Math.max(280, window.innerHeight - 250)
  )
  const measure = useCallback((index: number, next: number) => {
    if (heights.current.get(index) !== next) {
      heights.current.set(index, next)
      list.current?.resetAfterIndex(index)
    }
  }, [])
  const data = useMemo(() => ({ render, measure }), [render, measure])
  useLayoutEffect(() => {
    heights.current.clear()
    list.current?.resetAfterIndex(0)
  }, [ids])
  useEffect(() => {
    let width = container.current?.getBoundingClientRect().width
    const resize = () => {
      const nextWidth = container.current?.getBoundingClientRect().width
      if (nextWidth !== width) {
        width = nextWidth
        heights.current.clear()
        list.current?.resetAfterIndex(0)
      }
      setHeight(Math.max(280, window.innerHeight - 250))
    }
    window.addEventListener("resize", resize)
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize)
    if (container.current) observer?.observe(container.current)
    return () => {
      window.removeEventListener("resize", resize)
      observer?.disconnect()
    }
  }, [])
  return (
    <div
      ref={container}
      role="region"
      aria-label="Duplicate groups"
      data-testid="virtual-groups">
      <VariableSizeList
        ref={list}
        width="100%"
        height={height}
        itemCount={ids.length}
        itemSize={(index) => heights.current.get(index) ?? 340}
        estimatedItemSize={340}
        itemData={data}
        itemKey={(index) => ids[index]}
        overscanCount={3}>
        {MeasuredRow}
      </VariableSizeList>
    </div>
  )
}
