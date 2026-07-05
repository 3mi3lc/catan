// The transient dice moment shared by the hotseat and online apps. The
// vanilla clients restarted the CSS animation by removing `.show`, forcing a
// reflow, and re-adding it; in React the same restart falls out of keying
// the element by a per-roll nonce — a new roll remounts the node, replaying
// `diceIn` from the top. The overlay stays mounted (and invisible — the
// animation ends at opacity 0) between rolls, exactly like the vanilla one.

const PIPS: Record<number, [number, number][]> = {
    1: [[27, 27]],
    2: [[16, 16], [38, 38]],
    3: [[16, 16], [27, 27], [38, 38]],
    4: [[16, 16], [38, 16], [16, 38], [38, 38]],
    5: [[16, 16], [38, 16], [27, 27], [16, 38], [38, 38]],
    6: [[16, 14], [38, 14], [16, 27], [38, 27], [16, 40], [38, 40]],
};

function DieFace({ n }: { n: number }) {
    return (
        <div className="die">
            <svg viewBox="0 0 54 54">
                {(PIPS[n] ?? []).map(([x, y], i) => <circle key={i} cx={x} cy={y} r={4.6} fill="#3b3026" />)}
            </svg>
        </div>
    );
}

export interface DiceRoll { a: number; b: number; nonce: number }

export default function DiceOverlay({ roll }: { roll: DiceRoll | null }) {
    if (!roll) return <div id="dice" />;
    return (
        <div id="dice" className="show" key={roll.nonce}>
            <DieFace n={roll.a} />
            <DieFace n={roll.b} />
        </div>
    );
}
