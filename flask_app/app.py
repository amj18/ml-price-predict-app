from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import yaml

app = Flask(__name__)

# Configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html'), 200


@app.route('/output', methods=['POST'])
def user_form():
    """
    Allows user to create, update and delete recipes
    """
    if request.method == 'POST':
        form_details = request.form
        recipe_name = form_details['recipe_name']
        recipe_info = form_details['recipe_info']
        recipe_ingredients = form_details['recipe_ing']
        cur = mysql.connection.cursor()
        if request.form['submit_button'] == 'Create New Recipe':
            cur.execute(
                """
                INSERT INTO
                recipes(recipe_name, recipe_info, recipe_ingredients)
                VALUES(%s, %s, %s);
                """, (recipe_name,
                      recipe_info,
                      recipe_ingredients))
            results = cur.execute(
                """
                SELECT * FROM recipes
                ORDER BY date_created DESC
                LIMIT 1;
                """)
            mysql.connection.commit()
            return render_template('recipes.html',
                                   form_details=cur.fetchall()), 200
        elif request.form['submit_button'] == 'Update Recipe':
            results = cur.execute(
                            """
                            SELECT * FROM recipes
                            WHERE recipe_name=%s;
                            """, (recipe_name, ))
            if results == 0:
                return "No records found", 204
            else:
                cur.execute(
                    """
                    UPDATE recipes
                    SET recipe_info=%s, recipe_ingredients=%s
                    WHERE recipe_name=%s;
                    """, (recipe_info,
                          recipe_ingredients,
                          recipe_name))
                cur.execute(
                    """
                    SELECT * FROM recipes
                    ORDER BY date_created DESC
                    LIMIT 1;
                    """)
                mysql.connection.commit()
                return render_template('recipes.html',
                                       form_details=cur.fetchall()), 200
        elif request.form['submit_button'] == 'Delete Recipe':
            rows_deleted = cur.execute(
                                    """
                                    DELETE FROM recipes
                                    WHERE recipe_name=%s;
                                    """, (recipe_name,))
            mysql.connection.commit()
            return "{} recipe(s) with '{}' deleted.".format(str(rows_deleted),
                                                            recipe_name), 200
        else:
            return "Page not found. Check input parameters", 404


# Perhaps create a login here so only admin can view this?
@app.route('/recipes', methods=['GET'])
def recipes():
    """
    Allows user to view all recipes in the database.
    """
    cur = mysql.connection.cursor()
    results = cur.execute("SELECT * FROM recipes")
    if results > 0:
        form_details = cur.fetchall()
        return render_template('recipes.html', form_details=form_details), 200
    else:
        return "No entries in database!", 204
    cur.close()


@app.route('/recipes', methods=['POST'])
def search_recipe_by_ingredient():
    """
    Allows user to search for a recipe by ingredient.
    """
    form_details = request.form
    search_ingredients = form_details['recipe_ing_search']
    print(search_ingredients)
    cur = mysql.connection.cursor()
    if request.form['submit_button'] == 'Search':
        results = cur.execute(
            """
            SELECT * FROM recipes
            WHERE recipe_ingredients LIKE  %s
            """, ("%{}%".format(search_ingredients),))
        if results > 0:
            form_details = cur.fetchall()
            return render_template('recipes.html',
                                   form_details=form_details), 200
        else:
            return "No entries in database!", 204
        cur.close()


if __name__ == '__main__':
    app.run(debug=True)
