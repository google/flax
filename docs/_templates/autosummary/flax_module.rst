{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members:

   .. automethod:: __call__

   {% block methods %}

   {% for item in methods %}
   {%- if item not in inherited_members and item not in annotations and not item in ['__init__', 'setup'] %}
   .. automethod:: {{ item }}
   {%- endif %}
   {%- endfor %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::

   {% for item in methods %}
   {%- if item not in inherited_members and item not in annotations and not item in ['__init__', 'setup'] %}
       ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}