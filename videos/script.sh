#!/usr/bin/env bash
# Print the speaker script: one block per slide, in order, with timings.
#
# Everything comes out of presentation.html itself (data-say, data-dur,
# data-voice), so editing the deck updates the script -- there is no second copy
# to keep in sync.
#
# Usage:
#   ./script.sh           # read it in the terminal
#   ./script.sh --plain   # no colour, for piping to a file or printing
set -euo pipefail
cd "$(dirname "$0")"
[[ "${1:-}" == "--plain" ]] && NOCOLOR=1 || NOCOLOR=""
NOCOLOR="$NOCOLOR" python3 - presentation.html <<'PY'
import re, sys, os, html, textwrap

doc = open(sys.argv[1], encoding='utf-8').read()
plain = bool(os.environ.get('NOCOLOR'))
def c(code, t): return t if plain else f'\033[{code}m{t}\033[0m'

# Each <section class="slide ..."> is one beat of the talk.
sections = re.findall(r'<section class="slide[^>]*>', doc)
bodies   = re.split(r'<section class="slide[^>]*>', doc)[1:]

total = 0.0
blocks = []
for head, body in zip(sections, bodies):
    body = body.split('</section>')[0]
    dur  = float((re.search(r'data-dur="([\d.]+)"', head) or [0,'0'])[1])
    say  = re.search(r'data-say="([^"]*)"', head)
    say  = html.unescape(say.group(1)) if say else ''
    male = 'data-voice="male"' in head

    def clean(t):
        return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', t))).strip()

    # On-screen text, so you can see what the audience is reading as you talk.
    # Captions inside .swapgroup appear partway through, so they are marked
    # 'then:' -- do not read them out until the slide has swapped.
    onscreen = []
    swapgroup = ''
    gm = re.search(r'<div class="swapgroup">(.*?)</div>', body, re.S)
    if gm: swapgroup = gm.group(1)
    def later(frag): return bool(swapgroup) and frag in swapgroup

    for m in re.finditer(r'(<p[^>]*data-type="([^"]*)"[^>]*>)', body):
        txt = clean(m.group(2).replace('|', ''))
        onscreen.append(('then: ' if later(m.group(1)) else '') + txt)
    for m in re.finditer(r'(<p class="sub[^"]*">(.*?)</p>)', body, re.S):
        onscreen.append(('then: ' if later(m.group(1)) else '') + clean(m.group(2)))
    for m in re.finditer(r'<div class="stat[^"]*"><b>(.*?)</b><span>(.*?)</span>', body, re.S):
        onscreen.append(f'[stat] {clean(m.group(1))} — {clean(m.group(2))}')

    # Chart slides: the numbers are the point, so list every bar, not just the
    # heading -- these are the slides you are most likely to be asked about.
    # Split on the chart headings rather than trying to balance nested </div>s.
    chunks = re.split(r'(?=<h3>)', body)
    for ch in chunks:
        h3 = re.match(r'<h3>(.*?)</h3>', ch, re.S)
        if not h3: continue
        unit = re.search(r'<div class="unit">(.*?)</div>', ch, re.S)
        onscreen.append('[chart] ' + clean(h3.group(1)) + (f'  ({clean(unit.group(1))})' if unit else ''))
        for r in re.finditer(r'<span class="dev">(.*?)</span>.*?<span class="val">(.*?)</span>', ch, re.S):
            onscreen.append(f'         {clean(r.group(1)):>5} : {clean(r.group(2))}')
    for m in re.finditer(r'<p class="verdict">(.*?)</p>', body, re.S):
        onscreen.append('[verdict] ' + clean(m.group(1)))

    first = [o for o in onscreen if o and not o.startswith('then: ')]
    then_ = [o[6:] for o in onscreen if o.startswith('then: ')]
    blocks.append((total, dur, say, male, first, then_))
    total += dur + 0.6

w = 74
print()
print(c('1;33', f'  SPEAKER SCRIPT — {len(blocks)} slides, {int(total//60)}m {int(total%60):02d}s'))
print(c('2',    '  ' + '─'*w))
for i, (at, dur, say, male, onscreen, then_) in enumerate(blocks):
    stamp = f'{int(at//60)}:{int(at%60):02d}'
    tag = c('1;36', f'  {stamp}') + c('2', f'  slide {i}/{len(blocks)-1} · {dur:g}s')
    if male: tag += c('35', '  [male voice]')
    print()
    print(tag)
    if say:
        for line in textwrap.wrap(say, w-4):
            print('    ' + c('1;37', line))
    else:
        print('    ' + c('2', '(no narration — let the footage play)'))
    for o in onscreen:
        for j, line in enumerate(textwrap.wrap(o, w-8)):
            print('      ' + c('2', ('on screen: ' if j == 0 else '           ') + line))
    if then_:
        print('      ' + c('33', f'--- swaps at {dur*0.55:.0f}s ---'))
        for o in then_:
            for j, line in enumerate(textwrap.wrap(o, w-8)):
                print('      ' + c('2', ('then:      ' if j == 0 else '           ') + line))
print()
print(c('2', '  ' + '─'*w))
print(c('2', '  keys: →/space next · ← back · P autoplay · A narration · H hide HUD · F fullscreen'))
print()
PY
