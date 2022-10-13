{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members:

   {% block methods %}

   .. automethod:: __call__

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::

   {% for item in methods %}
   {%- if item not in inherited_members and item not in annotations and not item in ['__init__'] %}
       ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}