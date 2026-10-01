-- Point relative links at the rendered .html siblings instead of the .md sources.
function Link(el)
  if not el.target:match("^%a+://") then
    el.target = el.target:gsub("%.md(#?.*)$", ".html%1")
  end
  return el
end
