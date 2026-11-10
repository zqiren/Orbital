/**
 * Batch 1110 item 2: the goals card's 需要改 / "Needs changes" option used to
 * be sent bare, so the worker spent a turn asking what was wrong and the
 * correction was a second turn. A tap now prefills the reply with the option
 * and a colon; the words typed after it are the correction, and the daemon
 * still classifies the whole line as the guarded "change" answer.
 */
export function isChangeOption(option: string, changeOption: string | null | undefined): boolean {
  return !!changeOption && option === changeOption;
}

/** `需要改：` for a CJK option, `Needs changes: ` otherwise. */
export function changeOptionPrefill(option: string): string {
  return /[\u3400-\u9fff]/.test(option) ? `${option}：` : `${option}: `;
}
