FACE_STUDIO_CSS = """
:root {
  color-scheme: dark;
}

.gradio-container,
body {
  background: #120e1c;
}

.gradio-container {
  min-height: 100vh;
}

.gradio-container .block {
  background: #1a1328;
}

.gradio-container label.float {
  border-radius: 7px;
  background: rgba(24, 24, 24, 0.6);
  backdrop-filter: blur(20px);
  box-shadow: none;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.gradio-container label > span[data-testid="block-info"],
.gradio-container .block span.has-info {
  border: none;
  background: none;
  box-shadow: none;
  padding: 0;
}

.gallery-item .thumbnail-item {
  border-radius: 7px;
}

label.container.show_textbox_border textarea {
  background: #171026;
}

.radio_inline .wrap label > input[type="radio"] {
  display: none;
}

/* track */
fieldset.radio_inline > .wrap{
  display:inline-flex;
  gap:8px;
  padding:2px;
  border-radius:10px;
  background: var(--input-background-fill);
}

/* segment (text lives in the span) */
.radio_inline label, .radio_inline label.selected, .radio_inline label:hover {
  cursor:pointer;
  display:inline-flex;
  background:transparent;
  padding:0;
  border:none;
}
.radio_inline label > span{
  margin: 0;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  /* min-width:72px; */
  padding:8px 16px;
  border-radius:8px;
  font-weight:600;
  font-size:14px;
  line-height:1;
  color:#a3a8b3;                 /* inactive text */
  transition:
    background-color .18s ease,
    color .18s ease,
    box-shadow .18s ease,
    transform .06s ease;
}

.radio_inline label.selected > span{
  background:#2c174a;            /* deep purple chip */
  color:#d6c6ff;                 /* lavender text */
  box-shadow:inset 0 0 0 1px rgba(0,0,0,.25);
}

.radio_inline label:not(.selected):hover > span{
  color:#ded7f5;
}

/* Primary button neon green to pink gradient */
.gradio-container button.lg.primary {
  background-image: linear-gradient(90deg, #8f5bff 0%, #c06bff 100%);
  color: #0b0f13;
  border-radius: 10px;
}

.gradio-container button.lg.primary:hover {
  filter: saturate(1.1) brightness(1.2);
}

.gradio-container button.lg.primary:active {
  filter: saturate(1.1) brightness(0.8);
}

/* Match rounded corners across non-primary buttons */
.gradio-container button,
.gradio-container button.secondary,
.gradio-container button.stop,
.gradio-container button.sm,
.gradio-container button.md,
.gradio-container button.lg {
  border-radius: 10px;
}

"""
