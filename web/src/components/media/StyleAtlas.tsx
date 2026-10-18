import { useMemo, useState } from "react";
import {
  filterStyles,
  STYLE_ATLAS,
  styleThumbUrl,
  type ApplyStyleOptions,
  type StyleEntry,
} from "../../styleAtlas";

const PAGE = 48;

type Props = {
  disabled?: boolean;
  activeId?: string | null;
  onApply: (style: StyleEntry, options?: ApplyStyleOptions) => void;
};

export function StyleAtlas({ disabled, activeId, onApply }: Props) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [limit, setLimit] = useState(PAGE);
  const [openId, setOpenId] = useState<string | null>(null);
  /** Optional Ref2VA attach — remembered for grid + detail (+). */
  const [attachImage, setAttachImage] = useState(true);

  const rows = useMemo(() => filterStyles(query, category), [query, category]);
  const visible = rows.slice(0, limit);
  const open = openId
    ? rows.find((s) => s.id === openId) ?? STYLE_ATLAS.styles.find((s) => s.id === openId)
    : null;

  function apply(style: StyleEntry) {
    onApply(style, { attachImage });
  }

  if (open) {
    return (
      <div className="style-atlas style-atlas--detail">
        <div className="style-atlas__detail-bar">
          <button type="button" className="style-atlas__back" onClick={() => setOpenId(null)}>
            ← Looks
          </button>
          <button
            type="button"
            className="style-atlas__add"
            disabled={disabled}
            title={
              attachImage
                ? "Add look to prompt and attach preview as Ref2VA image"
                : "Add look to prompt"
            }
            onClick={() => apply(open)}
          >
            +
          </button>
        </div>
        <div className="style-atlas__hero-wrap">
          <img className="style-atlas__hero" src={styleThumbUrl(open.id)} alt="" />
        </div>
        <label className="style-atlas__attach">
          <input
            type="checkbox"
            checked={attachImage}
            disabled={disabled}
            onChange={(e) => setAttachImage(e.target.checked)}
          />
          <span>Also attach preview as Ref2VA image</span>
        </label>
        <p className="style-atlas__cat">{open.category}</p>
        <p className="style-atlas__full">{open.text}</p>
        {open.caption && open.caption !== open.text && (
          <p className="style-atlas__caption">{open.caption}</p>
        )}
        <p className="style-atlas__credit">
          Style Atlas by hoodtronik · {STYLE_ATLAS.dataset} by ostris
        </p>
      </div>
    );
  }

  return (
    <div className="style-atlas">
      <div className="style-atlas__filters">
        <input
          type="search"
          className="style-atlas__search"
          placeholder="Search looks…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setLimit(PAGE);
          }}
          disabled={disabled}
        />
        <select
          className="style-atlas__shelf"
          value={category}
          onChange={(e) => {
            setCategory(e.target.value);
            setLimit(PAGE);
          }}
          disabled={disabled}
          aria-label="Style category"
        >
          <option value="">All</option>
          {STYLE_ATLAS.categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>
      <label className="style-atlas__attach style-atlas__attach--bar">
        <input
          type="checkbox"
          checked={attachImage}
          disabled={disabled}
          onChange={(e) => setAttachImage(e.target.checked)}
        />
        <span>Attach look image as Ref2VA (+)</span>
      </label>
      <div className="style-atlas__grid">
        {visible.map((style) => (
          <div
            key={style.id}
            className={`style-atlas__cell${activeId === style.id ? " is-on" : ""}`}
          >
            <button
              type="button"
              className="style-atlas__thumb"
              disabled={disabled}
              title={style.lead}
              onClick={() => setOpenId(style.id)}
            >
              <img src={styleThumbUrl(style.id)} alt="" loading="lazy" />
            </button>
            <button
              type="button"
              className="style-atlas__add style-atlas__add--cell"
              disabled={disabled}
              title={
                attachImage
                  ? `Add ${style.lead} + Ref2VA image`
                  : `Add ${style.lead}`
              }
              onClick={() => apply(style)}
            >
              +
            </button>
          </div>
        ))}
      </div>
      {visible.length < rows.length && (
        <button type="button" className="style-atlas__more" onClick={() => setLimit((n) => n + PAGE)}>
          More ({rows.length - visible.length})
        </button>
      )}
    </div>
  );
}
