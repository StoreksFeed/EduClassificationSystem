from django import template

register = template.Library()

@register.simple_tag
def querystring(request, **kwargs):
    """
    Updates the querystring with the provided key-value pairs.
    """
    query_params = request.GET.copy()
    for key, value in kwargs.items():
        if value is None:
            query_params.pop(key, None)
        else:
            query_params[key] = value
    return query_params.urlencode()
