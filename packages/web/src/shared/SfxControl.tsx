// Mute button + volume slider, backed by audio/sound.ts's persisted
// settings. Same markup/ids as the vanilla clients' topbar control.

import { useState } from 'react';
import { isMuted, toggleMuted, getVolume, setVolume } from '../audio/sound';

export default function SfxControl() {
    const [muted, setMuted] = useState(isMuted());
    const [volume, setVolumeState] = useState(() => Math.round(getVolume() * 100));
    return (
        <div className="sfx-control">
            <button
                id="sfx-mute"
                className="ghost"
                title="Mute sound effects"
                onClick={() => { toggleMuted(); setMuted(isMuted()); }}
            >{muted ? '🔇' : '🔊'}</button>
            <input
                id="sfx-volume"
                type="range"
                min={0}
                max={100}
                value={volume}
                title="Sound volume"
                onChange={(e) => { const v = Number(e.target.value); setVolumeState(v); setVolume(v / 100); }}
            />
        </div>
    );
}
