// Transient bottom-center toast, shared by the hotseat and online apps.
// Same #toast markup and 2200ms lifetime as the vanilla implementation.

import { useCallback, useRef, useState } from 'react';

export function useToast(): { toastEl: React.ReactElement; toast: (msg: string) => void } {
    const [state, setState] = useState<{ msg: string; show: boolean }>({ msg: '', show: false });
    const timer = useRef<number | undefined>(undefined);

    const toast = useCallback((msg: string) => {
        setState({ msg, show: true });
        clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setState((s) => ({ ...s, show: false })), 2200);
    }, []);

    const toastEl = <div id="toast" className={state.show ? 'show' : ''}>{state.msg}</div>;
    return { toastEl, toast };
}
