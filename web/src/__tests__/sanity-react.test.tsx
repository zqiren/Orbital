import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

describe('react testing library sanity', () => {
  it('renders a trivial component and matches text', () => {
    render(<div>Hello</div>)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
