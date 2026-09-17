export const PROMPT_JSON_SCHEMA = JSON.stringify({
  type: 'object',
  properties: {
    tags: { type: 'array', items: { type: 'string' }, description: 'Danbooru/Gelbooru English tags' },
    caption: { type: 'string', description: 'At least two complete English sentences' },
    explanation_ko: { type: 'string', description: 'Brief Korean explanation, including the number of people' },
  },
  required: ['tags', 'caption', 'explanation_ko'],
  additionalProperties: false,
}, null, 2)

export function parseChatSchema(text: string): Record<string, unknown> {
  if (Array.from(text).length > 64000) throw Error('스키마는 64,000자 이하여야 합니다.')
  let schema: unknown
  try { schema = JSON.parse(text) } catch { throw Error('올바른 JSON을 입력하세요. 따옴표·쉼표·괄호를 확인해 주세요.') }
  if (!schema || typeof schema !== 'object' || Array.isArray(schema) || !Object.keys(schema).length) {
    throw Error('JSON 스키마는 비어 있지 않은 객체여야 합니다.')
  }
  return schema as Record<string, unknown>
}
