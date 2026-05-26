# Modified _merge_all_parts function to deduplicate sections
def _merge_all_parts(parts):
    seen_sections = set()
    merged_output = []
    for part in parts:
        for section in part:
            if section['title'] not in seen_sections:
                merged_output.append(section)
                seen_sections.add(section['title'])
    return merged_output

# Modified _generate_continuation function to keep track of written sections
def _generate_continuation(previous_part, current_part):
    written_sections = set([section['title'] for section in previous_part])
    continuation = []
    for section in current_part:
        if section['title'] not in written_sections:
            continuation.append(section)
    return continuation
